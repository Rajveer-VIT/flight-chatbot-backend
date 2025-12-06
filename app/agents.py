from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.tools import rag_search, search_flights, book_flight
import json
import re

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# CONSTANTS
# ------------------------------

ALLOWED_SMALL_TALK = [
    "hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening",
    "how are you", "how r u", "bye", "ok", "okay",
    "مرحبا", "شكرا", "السلام عليكم"
]

BLOCK_KEYWORDS = [
    "food", "recipe", "cook", "restaurant", "movie", "film",
    "song", "music", "politics", "prime minister", "pm", "weather",
    "football", "cricket", "sports", "math", "history", "science",
    "stock", "share", "bitcoin", "bank", "salary", "job"
]

FLIGHT_KEYWORDS = [
    "flight", "book", "booking", "pnr", "ticket", "airline", "baggage",
    "refund", "schedule", "airport", "departure", "arrival", "return",
    "cheap flights", "fare"
]

PERSONA_MESSAGE = """
You are FLIGHTBOT — a dedicated flight-booking assistant.
You ONLY answer flight-related questions.
Be friendly, short and accurate.
"""

# ------------------------------
# HELPERS
# ------------------------------

def detect_language(text: str) -> str:
    return "ar" if any("\u0600" <= ch <= "\u06FF" for ch in text) else "en"


# ------------------------------
# MAIN CHAT HANDLER
# ------------------------------

async def chatbot_reply(user_message: str, user_id: str):
    try:
        lang = detect_language(user_message)
        text = user_message.lower().strip()

        # ✅ 1️⃣ GREETINGS
        if text in ALLOWED_SMALL_TALK:
            return {
                "answer": "Hello! How can I help you with flights?"
                if lang == "en" else
                "مرحباً! كيف يمكنني مساعدتك في الرحلات؟",
                "source": "Greeting"
            }

        # ✅ 2️⃣ MANUAL CITY EXTRACTION (ROBUST)
        manual_match = re.search(r"from\s+(.+?)\s+to\s+(.+)", text)

        if manual_match:
            from_city = manual_match.group(1).strip()
            to_city = manual_match.group(2).strip()

            print("✅ MANUAL SEARCH:", from_city, "->", to_city)

            result = await search_flights({
                "from_city": from_city,
                "to_city": to_city
            })

            return {
                "answer": result,
                "source": "Manual-Search"
            }

        # ✅ 3️⃣ BLOCK NON-FLIGHT TOPICS
        if any(k in text for k in BLOCK_KEYWORDS) and not any(f in text for f in FLIGHT_KEYWORDS):
            return {
                "answer": "I'm sorry — I only help with flight booking, baggage, refunds and schedules."
                if lang == "en" else
                "عذراً — يمكنني فقط المساعدة في الرحلات.",
                "source": "Persona-Block"
            }

        # ✅ 4️⃣ RAG FAQ CHECK
        rag_result = rag_search(user_message)
        if rag_result:
            return {
                "answer": rag_result,
                "source": "RAG"
            }

        # ✅ 5️⃣ OPENAI TOOL FALLBACK (SAFE)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_flights",
                    "description": "Search flights using from and to cities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_city": {"type": "string"},
                            "to_city": {"type": "string"}
                        },
                        "required": ["from_city", "to_city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "book_flight",
                    "description": "Book a flight using flight ID and passenger name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "flight_id": {"type": "integer"},
                            "passenger_name": {"type": "string"}
                        },
                        "required": ["flight_id", "passenger_name"]
                    }
                }
            }
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": PERSONA_MESSAGE},
                {"role": "user", "content": user_message}
            ],
            tools=tools,
            tool_choice="auto"
        )

        msg = completion.choices[0].message

        # ✅ 6️⃣ SAFE TOOL HANDLER
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_call = msg.tool_calls[0]
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            if tool_name == "search_flights":
                result = await search_flights(args)
                return {"answer": result, "source": "AI-Tool"}

            if tool_name == "book_flight":
                result = await book_flight(args)
                return {"answer": result, "source": "Booking"}

        # ✅ 7️⃣ FINAL GPT FALLBACK
        return {
            "answer": msg.content or "Please tell me your travel route.",
            "source": "AI"
        }

    except Exception as e:
        print("❌ CHATBOT ERROR:", e)
        return {
            "answer": "Server error. Please try again in a moment.",
            "source": "Error"
        }
