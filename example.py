import torch

# Check Torch version
print("Torch version:", torch.__version__)

# Check if GPUs are available
if torch.cuda.is_available():
    print("GPUs are available")
else:
    print("GPUs are not available")


import random_tensor_laboratory as rtl

# list everything in rtl
print(dir(rtl))
