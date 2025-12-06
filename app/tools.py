import json, os, numpy as np, httpx, datetime, random, re
from openai import OpenAI
from app.config import OPENAI_API_KEY, FLIGHT_API_BASE_URL

client = OpenAI(api_key=OPENAI_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

# ✅ EMBEDDING
def create_embedding(text: str):
    e = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(e.data[0].embedding)

# ✅ LOAD FAQ
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    FAQ_DATA = json.load(f)

for faq in FAQ_DATA:
    faq["embedding_en"] = create_embedding(faq["question_EN"])
    faq["embedding_ar"] = create_embedding(faq["question_AR"])

def detect_language(text):
    return "ar" if re.search("[\u0600-\u06FF]", text) else "en"

def rag_search(query: str):
    lang = detect_language(query)
    query_emb = create_embedding(query)

    best_score = 0
    best_answer = None

    for faq in FAQ_DATA:
        faq_emb = faq["embedding_ar"] if lang == "ar" else faq["embedding_en"]

        similarity = np.dot(query_emb, faq_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(faq_emb)
        )

        if similarity > best_score:
            best_score = similarity
            best_answer = faq["answer_AR"] if lang == "ar" else faq["answer_EN"]

    if best_score >= 0.78:
        return best_answer

    return None

# ✅ CLOUD RUN SEARCH
async def search_flights(args: dict):
    url = f"{FLIGHT_API_BASE_URL}/flights/search"

    params = {
        "from": args["from_city"],
        "to": args["to_city"],
        "lang": "en"
    }

    async with httpx.AsyncClient(timeout=15) as client:
        res = await client.get(url, params=params)

    if res.status_code != 200:
        return {"error": "No flights found"}

    return {"flights": res.json()}

# ✅ MOCK BOOKING
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
