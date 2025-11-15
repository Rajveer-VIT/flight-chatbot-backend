from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.agents import chatbot_reply

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "Chatbot backend running"}

# --- WebSocket Chat API ---
@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    await websocket.accept()

    try:
        while True:
            user_message = await websocket.receive_text()
            
            # Generate chatbot reply using agents.py
            response = await chatbot_reply(user_message, user_id)

            await websocket.send_text(response)

    except WebSocketDisconnect:
        print(f"User {user_id} disconnected")
