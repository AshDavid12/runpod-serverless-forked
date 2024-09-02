# Include Python
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
RUN pip freeze > requirements_freeze_before.txt


# Install CUDA toolkit and specific CUDA libraries
RUN apt-get update && apt-get install -y cuda-toolkit-12-1 libcublas-12-1 libcudart-12-1 libcurand-12-1

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

RUN pip freeze > requirements_freeze_after.txt

# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
