import socket
import argparse
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Argument parser for host and port
parser = argparse.ArgumentParser(description='Client to send audio to the server.')
parser.add_argument('--host', type=str, default='localhost', help='The server host')
parser.add_argument('--port', type=int, default=43007, help='The server port')
args = parser.parse_args()

# Create a connection to the server
def connect_to_server(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        logger.info(f"Connected to server at {host}:{port}")
        return s
    except Exception as e:
        logger.error(f"Failed to connect to the server: {e}")
        sys.exit(1)

def main():
    host = args.host
    port = args.port

    # Connect to the server
    server_socket = connect_to_server(host, port)

    # Now the audio data will be piped in through the `rec` command via netcat
    logger.info("Waiting for audio data from rec and nc...")

    try:
        while True:
            data = server_socket.recv(4096)  # Adjust the buffer size as needed
            if not data:
                break
            logger.info(f"Received data: {len(data)} bytes")
    except KeyboardInterrupt:
        logger.info("Client interrupted by user.")
    finally:
        server_socket.close()

if __name__ == "__main__":
    main()
