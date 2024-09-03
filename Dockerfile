FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.11

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    libcublas11 \
    && rm -rf /var/lib/apt/lists/*


# Set up Python environment
RUN python3 -m pip install --upgrade pip

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install the specific model using faster-whisper
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

# Add your Python scripts
COPY infer.py .
COPY whisper_online.py .

# Run the infer.py script when the container starts
CMD ["python3", "-u", "/infer.py"]





# Include Python
#from python:3.11.1-buster
#
## Define your working directory
#WORKDIR /
#
## Install runpod
#COPY requirements.txt .
#RUN pip install -r requirements.txt
#
#RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'
#
## Add your file
#ADD infer.py .
#ADD whisper_online.py .
#
#ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib"
#
## Call your file when your container starts
#CMD [ "python", "-u", "/infer.py" ]