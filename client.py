import asyncio
import websockets
import json
import time

async def transcribe(audio_chunks):
    async with websockets.connect("ws://localhost:8000/ws/transcribe") as websocket:
        start_time = time.time()
        for i, chunk in enumerate(audio_chunks):
            end_time = start_time + 2.0  # Assume each chunk represents 2 seconds

            # WebSocket ensures that the entire chunk is sent and received as one atomic message.
            await websocket.send(chunk)

            # Receive the transcription response from the server
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Received transcription: {data['text']}")
            start_time = end_time

# Simulate 2-second audio chunks
async def main():
    audio_chunks = [b'fake_audio_data_chunk_1', b'fake_audio_data_chunk_2', b'fake_audio_data_chunk_3']
    await transcribe(audio_chunks)

asyncio.run(main())
