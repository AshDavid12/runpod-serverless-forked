# Include Python
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime@sha256:a355f16160f64219173261456bd5a62a8b99c3fb76ee405c7929a2c8df7dfeb3

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
RUN pip install -r requirements.txt

RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

# Add your file

# Diagnostic step to check for CUDA and cuDNN libraries
RUN echo "Checking for CUDA and cuDNN library paths..." \
    && find / -name "libcudnn*" -or -name "libcublas*" -or -name "libcuda*" \
    && echo "Completed check for CUDA/cuDNN libraries."

# Print the current LD_LIBRARY_PATH to verify its contents
RUN echo "LD_LIBRARY_PATH is set to: $LD_LIBRARY_PATH"

ENV LD_LIBRARY_PATH="/usr/local/lib/python3.9/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.9/site-packages/nvidia/cublas/lib"

ADD infer.py .
ADD whisper_online.py .


# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
