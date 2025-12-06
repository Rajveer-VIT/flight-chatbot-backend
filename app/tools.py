import json, os, numpy as np, httpx, datetime, random, re
from openai import OpenAI
from app.config import OPENAI_API_KEY, FLIGHT_API_BASE_URL

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

# -------------------------------
# 🔹 EMBEDDING (SAFE)
# -------------------------------
def create_embedding(text: str):
    try:
        e = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return e.data[0].embedding  # ✅ RETURN LIST (NOT numpy)
    except Exception as e:
        print("❌ EMBEDDING ERROR:", e)
        return []

# -------------------------------
# 🔹 LOAD FAQ + CACHE EMBEDDINGS
# -------------------------------
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)

updated = False

for faq in FAQ_DATA:
    if "embedding_en" not in faq:
        faq["embedding_en"] = create_embedding(faq["question_EN"])
        updated = True

    if "embedding_ar" not in faq:
        faq["embedding_ar"] = create_embedding(faq["question_AR"])
        updated = True

# ✅ SAVE EMBEDDINGS BACK TO FILE (VERY IMPORTANT)
if updated:
    with open(FAQ_PATH, "w", encoding="utf-8") as f:
        json.dump(FAQ_DATA, f, ensure_ascii=False, indent=2)

# -------------------------------
# 🔹 LANGUAGE DETECT
# -------------------------------
def detect_language(text: str):
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"

# -------------------------------
# 🔹 RAG SEARCH (OPTIMIZED)
# -------------------------------
def rag_search(query: str):
    lang = detect_language(query)
    query_emb = create_embedding(query)

    if not query_emb:
        return None

    query_emb = np.array(query_emb)

    best_score = 0
    best_answer = None

    for faq in FAQ_DATA:
        faq_emb = faq["embedding_ar"] if lang == "ar" else faq["embedding_en"]
        faq_emb = np.array(faq_emb)

        similarity = np.dot(query_emb, faq_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(faq_emb)
        )

        if similarity > best_score:
            best_score = similarity
            best_answer = faq["answer_AR"] if lang == "ar" else faq["answer_EN"]

    if best_score >= 0.78:
        return best_answer

    return None

# -------------------------------
# 🔹 CLOUD RUN FLIGHT SEARCH
# -------------------------------
async def search_flights(args: dict):
    try:
        url = f"{FLIGHT_API_BASE_URL}/flights/search"

        params = {
            "from": args.get("from_city"),
            "to": args.get("to_city"),
            "lang": "en"
        }

        async with httpx.AsyncClient(timeout=20) as client:
            res = await client.get(url, params=params)

        if res.status_code != 200:
            return {"error": "No flights found"}

        return {"flights": res.json()}

    except Exception as e:
        print("❌ SEARCH API ERROR:", e)
        return {"error": "Flight service unavailable"}

# -------------------------------
# 🔹 MOCK BOOKING (SAFE)
# -------------------------------
async def book_flight(args: dict):
    try:
        date_code = datetime.datetime.now().strftime("%d%m%y")
        pnr = f"FL-{date_code}-{random.randint(10000, 99999)}"

        return {
            "ticket": {
                "pnr": pnr,
                "flight_id": args.get("flight_id"),
                "passenger": args.get("passenger_name"),
                "booking_date": date_code,
                "status": "CONFIRMED"
            }
        }

    except Exception as e:
        print("❌ BOOKING ERROR:", e)
        return {"error": "Booking failed"}
