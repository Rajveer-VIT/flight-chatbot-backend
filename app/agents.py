from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.tools import rag_search, search_flights, book_flight
import json

client = OpenAI(api_key=OPENAI_API_KEY)

# Allowed general messages (won’t be blocked)
ALLOWED_SMALL_TALK = [
    "hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening",
    "how are you", "how r u", "bye", "ok", "okay", "مرحبا", "شكرا", "السلام عليكم"
]

# Strongly non-flight topics
BLOCK_KEYWORDS = [
    "food", "recipe", "cook", "restaurant", "movie", "film",
    "song", "music", "politics", "prime minister", "pm", "weather",
    "football", "cricket", "sports", "math", "history", "science",
    "stock", "share", "bitcoin", "bank", "salary", "job"
]

# Flight keywords — allow these always
FLIGHT_KEYWORDS = [
    "flight", "book", "booking", "pnr", "ticket", "airline", "baggage",
    "refund", "schedule", "airport", "departure", "arrival", "return",
    "cheap flights", "fare"
]

PERSONA_MESSAGE = """
You are FLIGHTBOT — a dedicated flight-booking assistant.
You ONLY answer flight-related questions: flight search, booking, baggage rules, refunds, airlines, airport info.
You DO NOT answer non-flight questions like food, sports, movies, politics, weather, general knowledge, math, science, etc.
Be friendly and short.
"""


def detect_language(text):
    return "ar" if any("\u0600" <= ch <= "\u06FF" for ch in text) else "en"


async def chatbot_reply(user_message: str, user_id: str):
    lang = detect_language(user_message)
    text = user_message.lower().strip()

    # 1️⃣ Allow small talk
    if any(text == g for g in ALLOWED_SMALL_TALK):
        return {
            "answer": "Hello! How can I help you with flights?" if lang=="en" else "مرحباً! كيف يمكنني مساعدتك في الرحلات؟",
            "source": "Greeting"
        }

    # 2️⃣ If clearly non-flight, BLOCK
    if any(k in text for k in BLOCK_KEYWORDS) and not any(f in text for f in FLIGHT_KEYWORDS):
        return {
            "answer": "I'm sorry — I only help with flight booking, baggage, refunds, schedules or travel-related queries."
                      if lang=="en" else
                      "عذراً — يمكنني فقط المساعدة في حجز الرحلات، الأمتعة، الاسترداد، والجداول.",
            "source": "Persona-Block"
        }

    # 3️⃣ RAG Check
    rag_result = rag_search(user_message)
    if rag_result:
        return {"answer": rag_result, "source": "RAG"}

    # 4️⃣ Define Tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search flights between two cities",
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
                "description": "Book a flight using its ID and passenger name",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "flight_id": {"type": "integer"},
                        "passenger_name": {"type": "string"}
                    },
                    "required": ["flight_id", "passenger_name"]
                }
            }
        },
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

    # 5️⃣ If tool is called by GPT
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        if tool_name == "search_flights":
            result = await search_flights(args)
            return {"tool_result": result}

        if tool_name == "book_flight":
            result = await book_flight(args)

            # 🛫 Format Ticket Output (Arabic + English)
            if lang == "ar":
                return {
                    "answer": f"""
🎫 **تم تأكيد الحجز بنجاح**

🪪 **PNR:** {result['ticket']['pnr']}
👤 **الراكب:** {result['ticket']['passenger']}
✈️ **رقم الرحلة:** {result['ticket']['flight_id']}
📆 **تاريخ الحجز:** {result['ticket']['booking_date']}
📍 **الحالة:** مؤكد

📧 سيتم إرسال التذكرة الإلكترونية قريباً.
""",
                    "source": "Booking"
                }
            else:
                return {
                    "answer": f"""
🎫 **Flight Ticket Confirmed**

🪪 **PNR:** {result['ticket']['pnr']}
👤 **Passenger:** {result['ticket']['passenger']}
✈️ **Flight ID:** {result['ticket']['flight_id']}
📆 **Booking Date:** {result['ticket']['booking_date']}
📍 **Status:** CONFIRMED

📧 You will receive the e-ticket shortly.
""",
                    "source": "Booking"
                }

    # 6️⃣ Fallback AI answer
    ai_text = msg.content or ""
    return {"answer": ai_text, "source": "AI"}
