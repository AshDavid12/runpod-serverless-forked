import runpod
import base64
from dotenv import load_dotenv
import os
import openai

load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
RUN_POD_API_KEY = os.getenv('RUN_POD_API_KEY')
RUNPOD_ENDPOINT_ID =os.getenv('RUNPOD_ENDPOINT_ID')


mp3_data = open('me-hebrew.wav', 'rb').read()
data = base64.b64encode(mp3_data).decode('utf-8')
payload = { 'type' : 'blob', 'data' : data }

runpod.api_key = RUN_POD_API_KEY
ep = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
res = ep.run_sync(payload)