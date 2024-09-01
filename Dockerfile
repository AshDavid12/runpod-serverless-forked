# Include Python
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
RUN pip install -r requirements.txt

RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

# Add your file
ADD infer.py .
ADD whisper_online.py .

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.9/site-packages/nvidia/cublas/lib"

# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
