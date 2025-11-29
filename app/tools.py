import json
import os
import numpy as np
from openai import OpenAI
from app.config import OPENAI_API_KEY, FLIGHT_API_BASE_URL
import httpx
import datetime
import random

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

# Load FAQ data
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)


# ================================
# 🔹 Create Embedding
# ================================
def create_embedding(text: str):
    e = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(e.data[0].embedding)


# ================================
# 🔹 Precompute FAQ Embeddings (💡)
# ================================
for faq in FAQ_DATA:
    if "embedding" not in faq:
        faq["embedding"] = create_embedding(faq["question"])


# ================================
# 🔹 Smart Semantic RAG Search 🚀
# ================================
def rag_search(query: str):
    query_emb = create_embedding(query)

    best_score = -1
    best_answer = None

    for faq in FAQ_DATA:
        similarity = np.dot(query_emb, faq["embedding"]) / (
            np.linalg.norm(query_emb) * np.linalg.norm(faq["embedding"])
        )

        if similarity > best_score:
            best_score = similarity
            best_answer = faq["answer"]

    # 🎯 Smart Decision Logic
    if best_score >= 0.82:
        return best_answer
    elif best_score >= 0.72:
        return best_answer + "\n\nℹ️ This answer is based on closest available information."
    else:
        return None



# =======================
# Flight Search
# =======================
async def search_flights(args: dict):
    url = f"{FLIGHT_API_BASE_URL}/flights/search?from={args['from_city']}&to={args['to_city']}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code != 200:
        return {"error": "No flights found"}

    return {"flights": response.json()}


# =======================
# Flight Booking
# =======================
async def book_flight(args: dict):
    date_code = datetime.datetime.now().strftime("%d%m%y")
    pnr = f"FL-{date_code}-{random.randint(10000, 99999)}"

    return {
        "ticket": {
            "pnr": pnr,
            "flight_id": args["flight_id"],
            "passenger": args["passenger_name"],
            "booking_date": date_code,
            "status": "CONFIRMED"
        }
    }
