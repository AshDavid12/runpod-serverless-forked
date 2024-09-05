import socket
import logging
import asyncio
import base64
import runpod
import aiohttp
import os
from dotenv import load_dotenv
import sys

host = 'localhost'
port = 43007
SAMPLING_RATE = 16000
CHUNK_SIZE = 4096  # Size of audio chunks received over the socket

# Initialize logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Load environment variables
logging.info("Loading environment variables from .env file")
load_dotenv('.env')

# Retrieve API keys from environment variables
RUNPOD_ENDPOINT_ID_B = os.getenv('RUNPOD_ENDPOINT_ID_B')
RUN_POD_API_KEY = os.getenv('RUN_POD_API_KEY')
# Set the Runpod API key
runpod.api_key = RUN_POD_API_KEY
logging.info("Runpod API key set successfully")
if not RUN_POD_API_KEY or not RUNPOD_ENDPOINT_ID_B:
    logging.error("One or more environment variables are missing.")
    exit(1)


# Define the Connection class for handling socket communication
class Connection:
    '''Handles TCP communication with the client'''
    def __init__(self, conn):
        self.conn = conn
        self.conn.setblocking(True)

    def receive_audio_chunk(self):
        '''Receives raw audio chunks from the client'''
        try:
            return self.conn.recv(CHUNK_SIZE)
        except ConnectionResetError:
            return None


# Define the ServerProcessor class to handle incoming audio chunks
class ServerProcessor:
    def __init__(self, connection):
        self.connection = connection
        self.last_end = None

    def process(self):
        '''Processes audio chunks from the client'''
        while True:
            audio_chunk = self.connection.receive_audio_chunk()
            if not audio_chunk:
                break
            logger.info(f"Received audio chunk of size: {len(audio_chunk)} bytes")

            # Forward the chunk to an external service (e.g., Runpod API)
            asyncio.run(self.forward_audio_chunk_to_service(audio_chunk))

    async def forward_audio_chunk_to_service(self, audio_chunk):
        '''Forward the audio chunk to an external transcription service using Runpod's AsyncioEndpoint'''
        # Convert audio chunk to base64
        audio_base64 = base64.b64encode(audio_chunk).decode('utf-8')

        # Payload to send to the service
        payload = {
            "input": {
                "type": "blob",
                "data": audio_base64
            }
        }

        # Set up an aiohttp session with a TCP connector to disable SSL if needed
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
            try:
                try:
                    logging.info("Initializing Runpod AsyncioEndpoint")
                    endpoint = runpod.AsyncioEndpoint(RUNPOD_ENDPOINT_ID_B, session)
                    logging.info("Runpod AsyncioEndpoint initialized successfully")
                except Exception as e:
                    logging.error(f"Error initializing Runpod AsyncioEndpoint: {e}")
                    return

                try:
                    logging.info("Starting Runpod job asynchronously")
                    job = await endpoint.run(payload)
                    logging.info("Runpod job started successfully")
                except Exception as e:
                    logging.error(f"Error starting Runpod job asynchronously: {e}")
                    return
                # Polling job status
                while True:
                    status = await job.status()
                    logger.info(f"Current job status: {status}")

                    if status == "COMPLETED":
                        output = await job.output()
                        logger.info(f"Job output: {output}")
                        self.send_result(output)
                        break  # Exit the loop once the job is completed

                    elif status in ["FAILED"]:
                        logger.error("Job failed or encountered an error.")
                        break

                    else:
                        logger.info("Job in queue or processing. Waiting 3 seconds...")
                        await asyncio.sleep(3)  # Wait for 3 seconds before polling again

            except Exception as e:
                logger.error(f"Error occurred while interacting with Runpod: {e}")

    def format_output_transcript(self, o):
        '''Format the output transcript based on the timestamp and text'''
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end
            logger.info(f"Formatted transcript: {beg} {end} {o[2]}")
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        '''Send the formatted transcription result back to the client'''
        msg = self.format_output_transcript(o)
        if msg is not None:
            logger.info(f"Sending result: {msg}")


# Define the server loop to handle incoming connections
def start_server():
    '''Starts the TCP server and listens for connections'''
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        logger.info(f"Server listening on {host}:{port}")

        while True:
            conn, addr = s.accept()
            logger.info(f"Connected to client at {addr}")
            connection = Connection(conn)
            processor = ServerProcessor(connection)
            processor.process()
            conn.close()
            logger.info('Connection to client closed')


if __name__ == "__main__":
    start_server()
