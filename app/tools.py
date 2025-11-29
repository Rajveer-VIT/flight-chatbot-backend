import json 
import os
import numpy as np
from openai import OpenAI
from app.config import OPENAI_API_KEY, FLIGHT_API_URL  # <--- IMPORTED CORRECTLY
import httpx
import datetime
import random

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

# Load FAQ data
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)


def create_embedding(text: str):
    e = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(e.data[0].embedding)


def rag_search(query: str):
    query_emb = create_embedding(query)
    best_score = -1
    best_answer = None

    for faq in FAQ_DATA:
        faq_emb = create_embedding(faq["question"])
        similarity = np.dot(query_emb, faq_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(faq_emb)
        )

        if similarity > best_score:
            best_score = similarity
            best_answer = faq["answer"]

    return best_answer if best_score >= 0.80 else None


# =======================
# Flight Search Tool
# =======================
async def search_flights(args: dict):
    url = f"{FLIGHT_API_URL}/flights/search?from={args['from_city']}&to={args['to_city']}"  # <--- FIXED

    async with httpx.AsyncClient() as client:
        response = await client.get(url)

    if response.status_code != 200:
        return {"error": "No flights found"}

    return {"flights": response.json()}


# =======================
# Flight Booking Tool
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
