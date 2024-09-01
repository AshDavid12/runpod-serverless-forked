# Include Python
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Define your working directory
WORKDIR /

COPY requirements.txt .
# Install runpod
RUN pip install -r requirements.txt


RUN python3 -c 'import faster_whisper; m = faster_whisper.WhisperModel("ivrit-ai/faster-whisper-v2-d3-e3")'

## Add diagnostic commands to find library locations
#RUN echo "Finding CUDA and cuDNN library paths..." \
#    && find / -name "libcudnn*" \
#    && find / -name "libcublas*" \
#    && find / -name "libcuda*" \
#    && echo "Finished finding library paths."

ENV CUDA_LIB_PATH1=/opt/conda/lib/python3.11/site-packages/torch/lib
ENV CUDA_LIB_PATH2=/opt/conda/pkgs/pytorch-2.4.0-py3.11_cuda12.1_cudnn9.1.0_0/lib/python3.11/site-packages/torch/lib
ENV CUDA_LIB_PATH3=/usr/local/cuda-12.1/targets/x86_64-linux/lib
ENV CUDA_LIB_PATH4=/opt/conda/lib
ENV CUDA_LIB_PATH5=/opt/conda/pkgs/libcublas-12.1.0.26-0/lib
ENV CUDA_LIB_PATH6=/usr/local/cuda-12.1/compat
ENV CUDA_LIB_PATH7=/opt/nvidia/nsight-compute/2023.1.1/target/linux-desktop-glibc_2_11_3-x64
ENV CUDA_LIB_PATH8=/opt/nvidia/nsight-compute/2023.1.1/target/linux-desktop-glibc_2_19_0-ppc64le
ENV CUDA_LIB_PATH9=/opt/nvidia/nsight-compute/2023.1.1/target/linux-desktop-t210-a64

ENV LD_LIBRARY_PATH=$CUDA_LIB_PATH1:$CUDA_LIB_PATH2:$CUDA_LIB_PATH3:$CUDA_LIB_PATH4:$CUDA_LIB_PATH5:$CUDA_LIB_PATH6:$CUDA_LIB_PATH7:$CUDA_LIB_PATH8:$CUDA_LIB_PATH9:$LD_LIBRARY_PATH


ADD infer.py .
ADD whisper_online.py .


# Call your file when your container starts
CMD [ "python", "-u", "/infer.py" ]
