# Include Python
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
RUN pip install -r requirements.txt


RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

# Add diagnostic commands to find library locations
RUN echo "Finding CUDA and cuDNN library paths..." \
    && find / -name "libcudnn*" \
    && find / -name "libcublas*" \
    && find / -name "libcuda*" \
    && echo "Finished finding library paths."


ADD infer.py .
ADD whisper_online.py .


# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
