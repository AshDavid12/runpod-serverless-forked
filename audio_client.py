import asyncio
import websockets
import subprocess

# Async audio source using 'rec'
async def your_audio_source():
    print("in the your audio source")
    process = subprocess.Popen(
        ['rec', '-r', '16000', '-b', '16', '-e', 'signed-integer', '--channels=1', '-t', 'raw', '-'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # Capture stderr for debugging
        bufsize=0  # Ensure no buffering
    )


    CHUNK = 1024
    try:
        while True:
            audio_block = process.stdout.read(CHUNK)
            if not audio_block:
                print("No audio block received, breaking...")
                break
            print(f"Audio block size: {len(audio_block)}")  # Print audio block size
            yield audio_block
            await asyncio.sleep(0)
    finally:
        process.terminate()

async def send_audio_loop(source, websocket):
    print("in the send audio loop")
    async for audio_block in source:
        await websocket.send(audio_block)

# Continuously receive messages from the WebSocket server
async def read_output_loop(websocket):
    async for message in websocket:
        print("in the read_output_loop")
        print(f"Received transcription: {message}")

# Run the client
async def main():
    print("in main")
    uri = "ws://localhost:8000/ws/transcribe"

    async with websockets.connect(uri) as websocket:
        # Replace 'source' with your actual audio data source (e.g., a generator).
        source = your_audio_source()  # This should be an async generator yielding audio blocks

        # Run the send and receive loops concurrently.
        send_task = asyncio.create_task(send_audio_loop(source, websocket))
        receive_task = asyncio.create_task(read_output_loop(websocket))

        # Wait for both tasks to complete (or handle them individually).
        await asyncio.gather(send_task, receive_task)


# Run the asyncio event loop
asyncio.run(main())

