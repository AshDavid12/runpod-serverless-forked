# diagnostics.py
import os
import torch

# 1. Check LD_LIBRARY_PATH and Other Environment Variables
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
print("CUDA_VERSION:", os.environ.get("CUDA_VERSION"))

print("CUDA is available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("CUDA Capability:", torch.cuda.get_device_capability(0))
    print("CUDA Device Count:", torch.cuda.device_count())
else:
    print("CUDA is not available.")

print("PyTorch version:", torch.__version__)
print("cuDNN version:", torch.backends.cudnn.version())

# 2. List Available CUDA and cuDNN Libraries
cuda_libs = [
    "libcudnn_ops_infer.so.8",
    "libcudnn_ops_train.so.8",
    "libcudnn.so.8"
]
for lib in cuda_libs:
    print(f"Searching for {lib}...")
    found = False
    for root, dirs, files in os.walk("/"):
        if lib in files:
            print(f"Found {lib} in {root}")
            found = True
            break
    if not found:
        print(f"{lib} not found.")

# 3. Test CUDA and cuDNN Usage
if torch.cuda.is_available():
    a = torch.randn((1000, 1000), device='cuda')
    b = torch.randn((1000, 1000), device='cuda')
    c = torch.matmul(a, b)
    print("Matrix multiplication test passed.")
else:
    print("CUDA is not available, cannot perform the test.")

print("cuDNN enabled:", torch.backends.cudnn.enabled)

# 4. Inspect Dynamic Linking
os.system("ldd $(which python3)")
