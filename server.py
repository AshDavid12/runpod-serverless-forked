import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from faker import Faker

fake = Faker()

app = FastAPI()

class TranscriptionResponse(BaseModel):
    start: float
    end: float
    text: str

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # WebSocket guarantees that this `receive_bytes()` call will return the full message that the client sent in one `send()` call.
            audio_chunk = await websocket.receive_bytes()

            # Simulate transcription (in reality, you would replace this with your transcription model)
            response = TranscriptionResponse(
                start=0.0,  # Use actual timestamps based on your audio chunks
                end=2.0,
                text=fake.sentence()
            )
            await asyncio.sleep(0.5)  # Simulate processing delay
            await websocket.send_json(response.dict())

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # No need to manually close the WebSocket here
        print("WebSocket connection closed")