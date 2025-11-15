from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.agents import chatbot_reply

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Chatbot backend running"}

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    await websocket.accept()

    while True:
        try:
            text = await websocket.receive_text()

            response = await chatbot_reply(text, user_id)

            await websocket.send_json(response)

        except Exception as e:
            print("WebSocket Error:", e)
            await websocket.close()
            break
