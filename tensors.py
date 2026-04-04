import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Check if GPU is available
print(f"PyTorch version: {torch.__version__}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ No GPU detected. Go to Runtime → Change runtime type → GPU")
    print("   (Some demos will still work, just slower)")

print("\n✅ Setup complete! Ready to go.")

