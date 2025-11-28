from openai import OpenAI
from app.config import OPENAI_API_KEY
from app.tools import rag_search, search_flights, book_flight
import json
import re

client = OpenAI(api_key=OPENAI_API_KEY)

# Language detection
def detect_language(text):
    return "ar" if re.search(r"[\u0600-\u06FF]", text) else "en"

# Allowed general messages
ALLOWED_SMALL_TALK = [
    "hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening",
    "how are you", "how r u", "bye", "ok", "okay", "مرحبا", "شكرا", "السلام عليكم"
]

# Block clearly unrelated topics
BLOCK_KEYWORDS = [
    "food", "recipe", "cook", "restaurant", "movie", "film",
    "song", "music", "politics", "bank", "weather", "football",
    "cricket", "sports", "math", "history", "science", "salary"
]

PERSONA_MESSAGE = """
You are FLIGHTBOT — a multilingual (English + Arabic) assistant specialized in:
✈️ Flight search, booking, ticket generation (PNR), baggage rules, refund help,
airport info, visa guidance, travel support.

🛑 You DO NOT answer general questions like sports, cooking, banking, news, politics, movies, math, or weather.

🌐 If user speaks Arabic → reply in Arabic.
🌐 If user speaks English → reply in English.

💡 Be professional, friendly, and informative.
"""


async def chatbot_reply(user_message: str, user_id: str):
    lang = detect_language(user_message)
    text = user_message.lower().strip()

    # 1️⃣ Greetings allowed
    if any(g in text for g in ALLOWED_SMALL_TALK):
        return {
            "answer": "👋 Hello! How can I help you with flights?" if lang == "en" 
                      else "👋 مرحباً! كيف يمكنني مساعدتك في حجز الرحلات؟",
            "source": "Greeting"
        }

    # 2️⃣ Block irrelevant queries
    if any(k in text for k in BLOCK_KEYWORDS):
        return {
            "answer": "❌ I only assist with flight booking, baggage, refunds, schedules, and travel help."
                       if lang == "en" else
                      "❌ يمكنني فقط المساعدة في حجز الرحلات، الأمتعة، الاسترداد، والجداول ودعم السفر.",
            "source": "Blocked"
        }

    # 3️⃣ RAG for FAQ
    faq_answer = rag_search(user_message)
    if faq_answer:
        return {"answer": faq_answer, "source": "RAG"}

    # 4️⃣ GPT Function Tools
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
                "description": "Book a flight and generate e-ticket",
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

    # 5️⃣ GPT with persona
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

    # 6️⃣ Handle tool calls
    if msg.tool_calls:
        tool = msg.tool_calls[0]
        args = json.loads(tool.function.arguments)

        if tool.function.name == "search_flights":
            result = await search_flights(args)
            return {"tool_result": result}

        if tool.function.name == "book_flight":
            result = await book_flight(args)

            if lang == "ar":
                return {
                    "answer": f"""
🎫 **تم تأكيد التذكرة بنجاح**

🪪 **رقم الحجز (PNR):** {result['ticket']['pnr']}
👤 **الراكب:** {result['ticket']['passenger']}
✈️ **رقم الرحلة:** {result['ticket']['flight_id']}
📆 **تاريخ الإصدار:** {result['ticket']['booking_date']}
📍 **الحالة:** مؤكد

✈️ سيتم إرسال التذكرة الإلكترونية قريباً.
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

    # 7️⃣ Fallback AI Answer
    return {"answer": msg.content or "I can help you with flights.", "source": "AI"}
