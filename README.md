# Whisper Streaming Ivrit-AI

## Project Overview

The Whisper Streaming Ivrit-AI project is a real-time transcription system built around the Ivrit-AI model for Hebrew transcription. This project leverages the faster-whisper model for fast and accurate speech recognition. On the server side, the ivrit-ai model is wrapped within the FasterWhisperASR class from the whisper-online project, enabling efficient asynchronous transcription. The client and server communicate via a Runpod endpoint, where the client sends WAV files, and the model processes the input and returns the transcription in real-time.

The project provides an asynchronous pipeline for handling transcription tasks using Runpod's asynchronous handling capabilities and NVIDIA CUDA for GPU acceleration. The server and client work together to process audio data, and the system is capable of handling large payloads up to 200MB.

### Key Features

- Ivrit-AI Model: The project utilizes the Ivrit-AI Hebrew transcription model, wrapped in FasterWhisperASR.
- Asynchronous Processing: Both the server and client run asynchronously, providing efficient real-time processing using Runpod.
- Runpod Integration: The system is deployed on Runpod with GPU acceleration to handle large transcription tasks.
- CUDA Support: NVIDIA CUDA is utilized for fast processing, allowing the system to handle real-time transcription efficiently.
- WAV Audio Input: The client sends WAV audio files to the server for transcription.
- Real-Time Transcription: Audio is streamed in chunks and transcribed asynchronously.

### Setup Instructions

Prerequisites
- Docker
- Runpod Account
- NVIDIA CUDA-compatible GPU
- Python 3.x
- GitHub Actions (for building the Docker image)
- Runpod API Key

#### Environment Variables
Create a .env file and include the following variables:

```bash
RUN_POD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
``` 

### Server-Side Setup

Docker Image: The project uses a Docker image based on an NVIDIA CUDA image to utilize GPU acceleration. The Dockerfile is provided in the repository, and the image is built using GitHub Actions.  

Build the Docker Image:
Clone the repository.
Build the Docker image using GitHub Actions or a similar CI/CD pipeline.
Deploy the Docker image on Runpod using the provided endpoint ID.
Running the Server: The server runs the infer.py script, which handles the transcription asynchronously.  

### Client-Side Setup
Install Dependencies:  
Clone the repository.  
Install the necessary Python packages using pip or Poetry.  
Run the Client: Use the following Python script (notebook.py) to send WAV files to the server for transcription. The client encodes the WAV file, sends it as a payload to the server, and receives the transcription asynchronously.
