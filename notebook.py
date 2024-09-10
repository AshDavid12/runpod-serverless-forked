import runpod
import base64
from dotenv import load_dotenv
import os
import openai
import logging
import sys
from runpod import AsyncioEndpoint,AsyncioJob
import asyncio
import aiohttp
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s',handlers=[logging.StreamHandler(sys.stdout)])
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
    logging.info("Reading audio file: test_hebrew.wav")
    mp3_data = open('me-hebrew.wav', 'rb').read()
    logging.info("Encoding audio file to base64")
    data = base64.b64encode(mp3_data).decode('utf-8')
    payload = {'type': 'blob', 'data': data, 'unique_id': str(time.time())}
    logging.info("Payload created successfully")
except Exception as e:
    logging.error(f"Error reading or encoding audio file: {e}")

# Set the Runpod API key
runpod.api_key = RUN_POD_API_KEY
logging.info("Runpod API key set successfully")


async def run_async_endpoint():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        try:
            logging.info("Initializing Runpod AsyncioEndpoint")
            endpoint = runpod.AsyncioEndpoint(RUNPOD_ENDPOINT_ID, session)
            logging.info("Runpod AsyncioEndpoint initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing Runpod AsyncioEndpoint: {e}")
            return

        try:
            logging.info("Starting Runpod job asynchronously")
            job = await endpoint.run(payload)
            # async for output in job.stream():
            #     print(f"here is output:{output}")
            # logging.info("Runpod job started successfully")
        except Exception as e:
            logging.error(f"Error starting Runpod job asynchronously: {e}")
            return

        # Polling job status
        while True:
            try:
                status = await job.status()
                logging.info(f"Current job status: {status}")

                if status == "COMPLETED":
                    try:
                        output = await job.output()
                        logging.info(f"Job output: {output}")
                    except Exception as e:
                        logging.error(f"Error retrieving job output: {e}")
                    break  # Exit the loop once the job is completed

                elif status in ["FAILED"]:
                    logging.error("Job failed or encountered an error.")
                    break

                else:
                    logging.info("Job in queue or processing. Waiting 3 seconds...")
                    await asyncio.sleep(3)  # Wait for 3 seconds before polling again

            except Exception as e:
                logging.error(f"Error polling job status: {e}")
                break


# Directly run the async function when the script is executed
asyncio.run(run_async_endpoint())
