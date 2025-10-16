from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import torch, firebase_admin
from firebase_admin import credentials, firestore

# ----------------------------
# Firebase + Model setup
# ----------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("FIREBASE_KEY.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()
# model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

app = Flask(__name__)

def get_user_location(uid):
    user_ref = db.collection("users").where("uid", "==", uid).limit(1).stream()
    for doc in user_ref:
        data = doc.to_dict()
        location_data = data.get("location", {})
        if isinstance(location_data, dict):
            address = location_data.get("address", "")
            city = location_data.get("city", "")
            province = location_data.get("province", "")
        else:
            address = location_data
            city = province = ""
        return {"address": address, "city": city, "province": province}
    raise ValueError(f"No user found with uid: {uid}")

import heapq

def hybrid_search(keyword, uid):
    user_loc = get_user_location(uid)
    user_city = user_loc.get("city", "").lower()
    user_province = user_loc.get("province", "").lower()
    query_text = f"{keyword} {user_city} {user_province}"
    query_emb = model.encode(query_text, convert_to_tensor=True)

    top_results = []  # min-heap to store top 10 results
    BATCH_SIZE = 50
    last_doc = None

    while True:
        query = db.collection("embeddings").limit(BATCH_SIZE)
        if last_doc:
            query = query.start_after(last_doc)
        batch = list(query.stream())
        if not batch:
            break

        for doc in batch:
            data = doc.to_dict()
            emb = torch.tensor(data["embedding"])
            similarity = util.cos_sim(query_emb, emb).item()
            crop_name = data.get("cropName", "")
            crop_match = 1.0 if keyword.lower() in crop_name.lower() else 0.0
            price = float(data.get("pricePerUnit", 1))
            price_score = 1 / (1 + (price / 100))

            farmer_id = data.get("farmerId")
            address = city = province = ""
            if farmer_id:
                farmer_ref = db.collection("users").document(farmer_id).get()
                if farmer_ref.exists:
                    farmer_data = farmer_ref.to_dict()
                    loc_data = farmer_data.get("location", {})
                    if isinstance(loc_data, dict):
                        address = loc_data.get("address", "")
                        city = loc_data.get("city", "")
                        province = loc_data.get("province", "")
                    else:
                        address = loc_data

            location_score = 0.0
            if city.lower() == user_city:
                location_score = 1.0
            elif province.lower() == user_province:
                location_score = 0.7

            total_score = (
                0.45 * similarity +
                0.20 * crop_match +
                0.15 * price_score +
                0.20 * location_score
            )

            result = {
                "listingId": data.get("listingId"),
                "cropName": crop_name,
                "price": price,
                "location": {"address": address, "city": city, "province": province},
                "similarity": similarity,
                "location_score": location_score,
                "total_score": total_score
            }

            # 🔹 keep top 10 only
            if len(top_results) < 10:
                heapq.heappush(top_results, (total_score, result))
            else:
                heapq.heappushpop(top_results, (total_score, result))

        last_doc = batch[-1]

    # return sorted top 10 descending
    return [r[1] for r in sorted(top_results, key=lambda x: -x[0])]

@app.route("/search", methods=["POST"])
def search():
    try:
        data = request.json
        keyword = data.get("keyword")
        uid = data.get("uid")
        if not keyword or not uid:
            return jsonify({"error": "Missing 'keyword' or 'uid'"}), 400
        results = hybrid_search(keyword, uid)
        return jsonify({"rankedListings": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 👇 For Vercel handler
def handler(event, context):
    return app(event, context)
