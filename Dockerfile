# Include Python
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
RUN pip install -r requirements.txt

# Copy diagnostics script into the Docker image
COPY diagnostics.py .

# Run diagnostics script to capture environment and library info
RUN python diagnostics.py
# Add diagnostic commands to find library locations
RUN echo "Finding CUDA and cuDNN library paths..." \
    && find / -name "libcudnn*" \
    && find / -name "libcublas*" \
    && find / -name "libcuda*" \
    && echo "Finished finding library paths."


RUN echo $LD_LIBRARY_PATH



# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
