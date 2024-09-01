import logging

import runpod

import base64
import faster_whisper
import tempfile
from whisper_online import *
#import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_name = 'ivrit-ai/faster-whisper-v2-d3-e3'
logging.info(f"Loading model: {model_name}")

model = faster_whisper.WhisperModel(model_name, device=device)
#model = FasterWhisperASR(model_name, modelsize=None, cache_dir=None, model_dir=None)
import requests

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
    logging.info("Starting transcription job")
    datatype = job['input'].get('type', None)
    if not datatype:
        logging.error("Datatype field not provided. Should be 'blob' or 'url'.")
        return {"error": "datatype field not provided. Should be 'blob' or 'url'."}

    if not datatype in ['blob', 'url']:
        logging.error(f"Invalid datatype: {datatype}")
        return {"error": f"datatype should be 'blob' or 'url', but is {datatype} instead."}

    # Get the API key from the job input
    api_key = job['input'].get('api_key', None)

    with tempfile.TemporaryDirectory() as d:
        audio_file = f'{d}/audio.mp3'

        if datatype == 'blob':
            logging.info("Decoding base64 audio data")
            mp3_bytes = base64.b64decode(job['input']['data'])
            open(audio_file, 'wb').write(mp3_bytes)
        elif datatype == 'url':
            success = download_file(job['input']['url'], MAX_PAYLOAD_SIZE, audio_file, api_key)
            if not success:
                return {"error": f"Error downloading data from {job['input']['url']}"}

        result = transcribe_core(audio_file)
        logging.info("Transcription completed")
        return {'result': result}


def transcribe_whisper_streaming(job):
    print("in transcribe_whisper_streaming")
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
    logging.info('Transcribing...')

    ret = {'segments': []}
    try:
        segs, dummy = model.transcribe(audio_file, init_prompt="")
        for s in segs:
            words = []
            for w in s.words:
                words.append({'start': w.start, 'end': w.end, 'word': w.word, 'probability': w.probability})

            seg = {'id': s.id, 'seek': s.seek, 'start': s.start, 'end': s.end, 'text': s.text, 'avg_logprob': s.avg_logprob,
                   'compression_ratio': s.compression_ratio, 'no_speech_prob': s.no_speech_prob, 'words': words}

            print(seg)
            logging.info(f"Processed segment: {seg}")
            ret['segments'].append(seg)
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
    return ret


runpod.serverless.start({"handler": transcribe})
