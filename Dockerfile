# Include Python
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime@sha256:a355f16160f64219173261456bd5a62a8b99c3fb76ee405c7929a2c8df7dfeb3

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
RUN pip install -r requirements.txt

# Ensure compatibility by force reinstalling the ctranslate2 version
RUN pip install --force-reinstall ctranslate2==3.24.0

RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

# Add your file

# Define separate environment variables for each path
ENV LIB_CUBLAS_PATH=/opt/conda/lib
ENV LIB_TORCH_PATH=/opt/conda/lib/python3.10/site-packages/torch/lib
ENV LIB_CTRANSLATE2_PATH=/opt/conda/lib/python3.10/site-packages/ctranslate2.libs
ENV LIB_CUBLAS_PKG_PATH=/opt/conda/pkgs/libcublas-11.11.3.6-0/lib
ENV LIB_CUDART_PKG_PATH=/opt/conda/pkgs/cuda-cudart-11.8.89-0/lib
ENV LIB_PYTORCH_PKG_PATH=/opt/conda/pkgs/pytorch-2.1.0-py3.10_cuda11.8_cudnn8.7.0_0/lib/python3.10/site-packages/torch/lib

# Combine these variables into LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=$LIB_CUBLAS_PATH:$LIB_TORCH_PATH:$LIB_CTRANSLATE2_PATH:$LIB_CUBLAS_PKG_PATH:$LIB_CUDART_PKG_PATH:$LIB_PYTORCH_PKG_PATH

ADD infer.py .
ADD whisper_online.py .


# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
