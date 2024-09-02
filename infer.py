import runpod

import base64
import faster_whisper
import tempfile
import logging
import torch
import sys
import requests

import whisper_online

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
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


def download_file(url, max_size_bytes, output_filename, api_key=None):
    """
    Download a file from a given URL with size limit and optional API key.

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

        # Send a GET request
        response = requests.get(url, stream=True, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad requests

        # Get the file size if possible
        file_size = int(response.headers.get('Content-Length', 0))

        if file_size > max_size_bytes:
            print(f"File size ({file_size} bytes) exceeds the maximum allowed size ({max_size_bytes} bytes).")
            return False

        # Download and write the file
        downloaded_size = 0
        with open(output_filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded_size += len(chunk)
                if downloaded_size > max_size_bytes:
                    print(f"Download stopped: Size limit exceeded ({max_size_bytes} bytes).")
                    return False
                file.write(chunk)

        print(f"File downloaded successfully: {output_filename}")
        return True

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        return False


def transcribe(job):
    datatype = job['input'].get('type', None)
    if not datatype:
        return {"error": "datatype field not provided. Should be 'blob' or 'url'."}

    if not datatype in ['blob', 'url']:
        return {"error": f"datatype should be 'blob' or 'url', but is {datatype} instead."}

    # Get the API key from the job input
    api_key = job['input'].get('api_key', None)

    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'

        if datatype == 'blob':
            mp3_bytes = base64.b64decode(job['input']['data'])
            open(audio_file, 'wb').write(mp3_bytes)
        elif datatype == 'url':
            success = download_file(job['input']['url'], MAX_PAYLOAD_SIZE, audio_file, api_key)
            if not success:
                return {"error": f"Error downloading data from {job['input']['url']}"}

        result = transcribe_core(audio_file)
        return {'result': result}


def transcribe_core(audio_file):
    print('Transcribing...')

    ret = {'segments': []}

    segs, dummy = model.transcribe(audio_file, language='he', word_timestamps=True)
    for s in segs:
        words = []
        for w in s.words:
            words.append({'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability})

        seg = {'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
               'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words}

        print(seg)
        ret['segments'].append(seg)

    return ret


#runpod.serverless.start({"handler": transcribe})

def transcribe_whisper(job):
    logging.info(f"in triscribe-whisper")
    datatype = job['input'].get('type', None)
    if not datatype:
        return {"error": "datatype field not provided. Should be 'blob' or 'url'."}

    if not datatype in ['blob', 'url']:
        return {"error": f"datatype should be 'blob' or 'url', but is {datatype} instead."}

    # Get the API key from the job input
    api_key = job['input'].get('api_key', None)

    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'

        if datatype == 'blob':
            mp3_bytes = base64.b64decode(job['input']['data'])
            open(audio_file, 'wb').write(mp3_bytes)
        elif datatype == 'url':
            success = download_file(job['input']['url'], MAX_PAYLOAD_SIZE, audio_file, api_key)
            if not success:
                return {"error": f"Error downloading data from {job['input']['url']}"}
        logging.info("Starting transcription process using transcribe_core_whisper.")
        result = transcribe_core_whisper(audio_file)
        logging.info(f"DONE: in triscribe-whisper")
        return {'result': result}

def transcribe_core_whisper(audio_file):
    print('Transcribing...')

    ret = {'segments': []}

    try:
        logging.debug(f"Transcribing audio file: {audio_file}")

        segs = model.transcribe(audio_file, init_prompt="")
        logging.info("Transcription completed successfully.")
        for s in segs:
            words = []
            for w in s.words:
                words.append({'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability})

            seg = {'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
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

runpod.serverless.start({"handler": transcribe_whisper})