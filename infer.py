import runpod

import base64
import faster_whisper
import tempfile
import logging
import torch
import sys
import requests
import asyncio
import whisper_online
import aiohttp

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])

# Try to import the module
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


async def download_file(url, max_size_bytes, output_filename, api_key=None):
    logging.info("infer.py-in download file async")
    """
    Asynchronously download a file from a given URL with size limit and optional API key.

    Args:
    url (str): The URL of the file to download.
    max_size_bytes (int): Maximum allowed file size in bytes.
    output_filename (str): The name of the file to save the download as.
    api_key (str, optional): API key to be used as a bearer token.

    Returns:
    bool: True if download was successful, False otherwise.
    """
    try:
        # Prepare headers
        headers = {}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'


        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()

                file_size = int(response.headers.get('Content-Length', 0))

                if file_size > max_size_bytes:
                    logging.warning(f"File size ({file_size} bytes) exceeds the maximum allowed size ({max_size_bytes} bytes).")
                    return False

                downloaded_size = 0
                with open(output_filename, 'wb') as file:
                    async for chunk in response.content.iter_chunked(8192):
                        downloaded_size += len(chunk)
                        if downloaded_size > max_size_bytes:
                            logging.warning(f"Download stopped: Size limit exceeded ({max_size_bytes} bytes).")
                            return False
                        file.write(chunk)

        logging.info(f"File downloaded successfully: {output_filename}")
        return True

    except aiohttp.ClientError as e:
        logging.error(f"Error downloading file: {e}")
        return False



import wave
import tempfile

def chunk_audio(file_path, chunk_size=1024):
    """ Yield chunks of audio data from a WAV file """
    with wave.open(file_path, 'rb') as wf:
        while True:
            data = wf.readframes(chunk_size)
            if not data:
                break  # End of file reached
            yield data
            # Write chunk to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(data)
                temp_file.flush()
                yield temp_file.name  # Yield the path to the temporary file


# Asynchronous function to handle the transcribe job
async def async_transcribe_whisper(job):
    logging.info("In async_transcribe_whisper")

    datatype = job['input'].get('type', None)
    if not datatype:
        yield {"error": "datatype field not provided. Should be 'blob' or 'url'."}
        return

    if datatype not in ['blob', 'url']:
        yield {"error": f"datatype should be 'blob' or 'url', but is {datatype} instead."}
        return

    api_key = job['input'].get('api_key', None)

    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'

        if datatype == 'blob':
            try:
                logging.info(f"type blob")
                mp3_bytes = base64.b64decode(job['input']['data'])
                with open(audio_file, 'wb') as f:
                    f.write(mp3_bytes)
            except Exception as e:
                logging.error(f"Error decoding blob data: {e}")
                yield {"error": str(e)}
                return

        elif datatype == 'url':
            success = await download_file(job['input']['url'], MAX_PAYLOAD_SIZE, audio_file, api_key)
            if not success:
                yield {"error": f"Error downloading data from {job['input']['url']}"}
                return

        logging.info("Starting transcription process using async_transcribe_core_whisper.")
        output_collected = False
        async for result in async_transcribe_core_whisper(audio_file):
            #logging.info(f"yielding transcription result:{result}")
            output_collected = True
            yield result

        if not output_collected:
            logging.warning("No transcription results were produced.")
        logging.info("DONE: in async_transcribe_whisper")


async def async_transcribe_core_whisper(audio_file):
    print('async trasncribe-core-whisper-Transcribing async...')

    ret = {'segments': []}

    try:
        logging.debug(f"Transcribing audio file: {audio_file}")
        for chunk in chunk_audio(audio_file):
            segs = await asyncio.to_thread(model.transcribe,chunk, init_prompt="")
            async for s in segs:
                logging.info(f"Segment type: {type(s)}; Segment details: {s}")
                if hasattr(s,'words'):
                    words = []
                    for w in s.words:
                        words.append({'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability})

                    seg = {'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
                           'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words}
                    logging.info(f"Processed segment: {seg}")
                    ret['segments'].append(seg)
                    # Log the current state of ret['segments']
                    logging.info(f"Current state of ret['segments']: {ret['segments']}")
                    yield {'result': seg}  # Yield each segment as it is processed
                else:
                    logging.warning(f"Segment has no 'words' attribute: {s}")
                    yield {"error": "Invalid segment format"}

    except Exception as e:
        logging.error(f"Error during async_transcribe_core_whisper: {e}", exc_info=True)
        yield {"error": str(e)}

    # Ensure final aggregation of all segments is yielded
    if ret['segments']:
        logging.info(f"Final transcription result: {ret}")
        yield {"final_result": ret}
    else:
        logging.warning("No segments to return in final result.")
    # Return the final result
    logging.info("Transcription core function completed.")
    sys.stdout.flush()



runpod.serverless.start({"handler": async_transcribe_whisper,
                         "return_aggregated_stream": True,
                         })