import asyncio
import logging
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from io import BytesIO
import whisper_online

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize FastAPI app
app = FastAPI()

try:
    logging.info("attempting to load whisper online")
    from whisper_online import *  # Replace 'some_module' with the actual module name

    logging.info("Successfully imported whisper_online.")
except ImportError as e:
    logging.error(f"Failed to import whisper_online: {e}", exc_info=True)
except Exception as e:
    logging.error(f"Unknown from exception- error to import whisper_online: {e}", exc_info=True)

if torch.cuda.is_available():
    logging.info(f"CUDA is available.")
else:
    logging.info("CUDA is not available. Using CPU.")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'ivrit-ai/faster-whisper-v2-d3-e3'
logging.info(f"Selected model name: {model_name}")
#model = faster_whisper.WhisperModel(model_name, device=device)
try:
    lan = 'he'
    logging.info(f"Attempting to initialize FasterWhisperASR with device: {device}")
    model = whisper_online.FasterWhisperASR(lan=lan, modelsize=model_name, cache_dir=None, model_dir=None)
    logging.info("FasterWhisperASR model initialized successfully.")
except Exception as e:
    logging.error(f"Falied to inilialize faster whisper model {e}")

# Maximum data size: 200MB
MAX_PAYLOAD_SIZE = 200 * 1024 * 1024


async def transcribe_audio(audio_file) -> dict:
    print('Transcribing...')

    ret = {'segments': []}

    try:
        logging.debug(f"Transcribing audio file: {audio_file}")
        audio_stream = BytesIO(audio_file)
        segs = model.transcribe(audio_stream, init_prompt="")
        logging.info("Transcription completed successfully.")
        for s in segs:
            words = []
            for w in s.words:
                words.append({'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability})

            seg = {'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text,
                   'avg_logprob': s.avg_logprob,
                   'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words}
            logging.debug(f"All segments processed. Final transcription result: {ret}")
            print(seg)
            ret['segments'].append(seg)

    except Exception as e:
        # Log any exception that occurs during the transcription process
        logging.error(f"Error during transcribe_core_whisper: {e}", exc_info=True)
        return {"error": str(e)}
    # Return the final result
    logging.info("Transcription core function completed.")
    return ret




@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    await websocket.accept()

    try:
        logging.info("WebSocket connection established.")
        audio_data = bytearray()

        # Continuously receive audio chunks
        while True:
            try:
                chunk = await websocket.receive_bytes()
                if chunk:
                    audio_data.extend(chunk)
                    logging.info(f"Received audio chunk of size {len(chunk)} bytes.")
            except WebSocketDisconnect:
                logging.info("WebSocket connection closed by the client.")
                break

        # Perform transcription directly from in-memory audio data
        result = await transcribe_audio(audio_data)

        # Send the transcription result back to the client
        await websocket.send_json(result)

    except Exception as e:
        logging.error(f"Error during WebSocket transcription: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        logging.info("Closing WebSocket connection.")
        await websocket.close()


# Run the server using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("audio_server:app", host="127.0.0.1", port=8000, reload=True)
