# Include Python
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
#RUN pip freeze > requirements_freeze_before.txt


# Install CUDA toolkit and specific CUDA libraries

RUN pip install -r requirements.txt
# Copy diagnostics script into the Docker image
#COPY diagnostics.py .
#
## Run diagnostics script to capture environment and library info
#RUN python diagnostics.py
# Add diagnostic commands to find library locations

# Set PATH and LD_LIBRARY_PATH
#ENV PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/opt/conda/pkgs/pytorch-2.3.1-py3.10_cuda12.1_cudnn8.9.2_0/lib/python3.10/site-packages/torch/lib:${PATH}"
#ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

RUN echo $LD_LIBRARY_PATH

RUN echo "Finding CUDA and cuDNN library paths..." \
    && find / -name "libcudnn*" \
    && find / -name "libcublas*" \
    && find / -name "libcuda*" \
    && echo "Finished finding library paths."


ADD infer.py .
ADD whisper_online.py .

#RUN pip freeze > requirements_freeze_after.txt

# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
