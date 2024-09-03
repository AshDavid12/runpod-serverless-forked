import runpod
import base64
from dotenv import load_dotenv
import os
import openai
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',handlers=[logging.StreamHandler(sys.stdout)])
# Set the logging level to DEBUG for more detailed output if needed
# logging.getLogger().setLevel(logging.DEBUG)

# Load environment variables from .env file
logging.info("Loading environment variables from .env file")
load_dotenv('.env')

# Retrieve API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
RUN_POD_API_KEY = os.getenv('RUN_POD_API_KEY')
RUNPOD_ENDPOINT_ID = os.getenv('RUNPOD_ENDPOINT_ID')
RUNPOD_ENDPOINT_ID_B = os.getenv('RUNPOD_ENDPOINT_ID_B')

# Log the status of loading environment variables
if not OPENAI_API_KEY or not RUN_POD_API_KEY or not RUNPOD_ENDPOINT_ID:
    logging.error("One or more environment variables are missing.")
else:
    logging.info("Environment variables loaded successfully.")

# Read and encode the audio file
try:
    logging.info("Reading audio file: me-hebrew.wav")
    mp3_data = open('me-hebrew.wav', 'rb').read()
    logging.info("Encoding audio file to base64")
    data = base64.b64encode(mp3_data).decode('utf-8')
    payload = {'type': 'blob', 'data': data}
    logging.info("Payload created successfully")
except Exception as e:
    logging.error(f"Error reading or encoding audio file: {e}")

# Set the Runpod API key
runpod.api_key = RUN_POD_API_KEY
logging.info("Runpod API key set successfully")

# Initialize the Runpod endpoint
try:
    logging.info("Initializing Runpod endpoint")
    ep = runpod.Endpoint(RUNPOD_ENDPOINT_ID)
    logging.info("Runpod endpoint initialized successfully")
except Exception as e:
    logging.error(f"Error initializing Runpod endpoint: {e}")

# Run the endpoint synchronously with the payload
try:
    logging.info("Running Runpod endpoint with payload")
    res = ep.run(payload)
    logging.info(f"Runpod endpoint executed successfully. Response: {res}")
except Exception as e:
    logging.error(f"Error running Runpod endpoint: {e}")
