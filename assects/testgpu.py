import torch

# This checks if Pylance successfully imports torch
print("--- Environment Check ---")
print("PyTorch Version:", torch.__version__)

# This checks if your graphics card is ready for CNN/GNN workloads
cuda_ready = torch.cuda.is_available()
print("CUDA Available:", cuda_ready)

if cuda_ready:
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print(
        "Warning: Running on CPU only. Check your PyTorch installation if you have an NVIDIA GPU."
    )
