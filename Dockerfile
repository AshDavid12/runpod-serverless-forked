# Include Python
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Define your working directory
WORKDIR /

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

# Install runpod
#RUN pip freeze > requirements_freeze_before.txt

# Install CUDA toolkit and specific CUDA libraries


# Copy diagnostics script into the Docker image
#COPY diagnostics.py .
#
## Run diagnostics script to capture environment and library info
#RUN python diagnostics.py
# Add diagnostic commands to find library locations

# Set PATH and LD_LIBRARY_PATH
#ENV PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/conda/pkgs/pytorch-2.3.1-py3.10_cuda12.1_cudnn8.9.2_0/lib/python3.10/site-packages/torch/lib:${PATH}"
#ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"


ADD infer.py .
ADD whisper_online.py .

#RUN pip freeze > requirements_freeze_after.txt

#ENV LD_LIBRARY_PATH="/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib"

# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
