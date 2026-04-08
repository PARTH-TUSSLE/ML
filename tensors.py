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



# TENSORS
#0D (scalars)
parth_age = torch.tensor(20.0)
print(parth_age)
# 1D (vectors)
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(vector)
print(vector.dim()) #->1
print(parth_age.dim()) #->0

# A scalar — just a number
scalar = torch.tensor(42.0)
print(f"Scalar: {scalar}")
print(f"  Shape: {scalar.shape}")      # torch.Size([])
print(f"  Dimensions: {scalar.dim()}")  # 0
print()

# A vector — a list of numbers
vector = torch.tensor([1.0, 2.0, 3.0, 4.0])
print(f"Vector: {vector}")
print(f"  Shape: {vector.shape}")      # torch.Size([4])
print(f"  Dimensions: {vector.dim()}")  # 1
print()

# A matrix — a grid of numbers
matrix = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
print(f"Matrix:\n{matrix}")
print(f"  Shape: {matrix.shape}")      # torch.Size([2, 3])
print(f"  Dimensions: {matrix.dim()}")  # 2
print()

# A 3D tensor — a cube of numbers
tensor_3d = torch.zeros(2, 3, 4)
print(f"3D Tensor shape: {tensor_3d.shape}")    # [2, 3, 4]
print(f"  Dimensions: {tensor_3d.dim()}")        # 3
print()

# A 4D tensor — like a batch going through multi-head attention
tensor_4d = torch.zeros(8, 12, 3, 64)
print(f"4D Tensor shape: {tensor_4d.shape}")    # [8, 12, 3, 64]
print(f"  That's: 8 sentences × 12 heads × 3 tokens × 64 dims")

# Zeros and ones
z = torch.zeros(3, 4)        # 3×4 matrix of zeros
o = torch.ones(2, 5)         # 2×5 matrix of ones

# Random (normal distribution) — this is how weights start!
r = torch.randn(3, 3)        # random values, mean=0, std=1
print("Random matrix (this is how neural network weights begin):")
print(r)
print()

# From a range
seq = torch.arange(0, 10)    # [0, 1, 2, ..., 9]
print(f"Sequence: {seq}")

# Linspace — evenly spaced
lin = torch.linspace(0, 1, 5)  # 5 values from 0 to 1
print(f"Linspace: {lin}")



# TRANSFORMATIONS


# A vector pointing right
v = torch.tensor([1.0, 0.0])
print(f"Original vector: {v.tolist()}")

# Rotation matrix (90 degrees counterclockwise)
rotate_90 = torch.tensor([[0., -1.],
                           [1.,  0.]])

rotated = rotate_90 @ v
print(f"After 90° rotation: {rotated.tolist()}")  # [0, 1] → pointing up!

# Rotation matrix (45 degrees)
angle = math.radians(45)
rotate_45 = torch.tensor([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle),  math.cos(angle)]])

rotated_45 = rotate_45 @ v
print(f"After 45° rotation: [{rotated_45[0]:.3f}, {rotated_45[1]:.3f}]") #Take this number and show it as a float with 3 decimal places



# DOT PRODUCT


# Two similar vectors (pointing roughly the same way)
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.1, 1.9, 3.1])
print(f"Similar vectors dot product: {a @ b:.2f}")  # high!

# Two perpendicular vectors
c = torch.tensor([1.0, 0.0])
d = torch.tensor([0.0, 1.0])
print(f"Perpendicular dot product:   {c @ d:.2f}")  # zero!

# Two opposite vectors
e = torch.tensor([1.0, 2.0])
f = torch.tensor([-1.0, -2.0])
print(f"Opposite vectors dot product: {e @ f:.2f}")  # negative!

print()
print("☝️ This is EXACTLY what attention does:")
print("   Q · K = how much should this word attend to that word?")



#PYTHON vs PYTORCH



# Multiply two matrices using Python loops (the slow way)
def matmul_slow(A, B):
    rows_A, cols_A = A.shape
    rows_B, cols_B = B.shape
    C = torch.zeros(rows_A, cols_B)
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

size = 100
A = torch.randn(size, size)
B = torch.randn(size, size)

# Slow way
start = time.time()
C_slow = matmul_slow(A, B)
slow_time = time.time() - start

# Fast way (PyTorch optimized)
start = time.time()
C_fast = A @ B
fast_time = time.time() - start

print(f"Matrix multiply {size}×{size}:")
print(f"  Python loops: {slow_time:.4f} seconds")
print(f"  PyTorch @:    {fast_time:.6f} seconds")
print(f"  Speedup:      {slow_time/fast_time:.0f}x faster! 🚀")
print()


#GPU Power


x = torch.randn(3, 4)
print(f"Shape: {x.shape}")
print(f"Data pointer: {x.data_ptr()}")

y = x.view(4, 3)
print(f"\nAfter view(4, 3):")
print(f"Shape: {y.shape}")
print(f"Data pointer: {y.data_ptr()}")  # SAME pointer!
print(f"\nSame memory? {x.data_ptr() == y.data_ptr()}")  # True!



#RESHAPING


# 1. Tensor internals
x = torch.randn(3, 4)
print(f"Shape: {x.shape}")
print(f"Stride: {x.stride()}")
print(f"Is contiguous: {x.is_contiguous()}")
print(f"Data pointer: {x.data_ptr()}")

y = x.view(4, 3)
print(f"\nReshaped — same memory? {x.data_ptr() == y.data_ptr()}")

z = x.T  # transpose
print(f"Transposed — contiguous? {z.is_contiguous()}")
print(f"Transposed stride: {z.stride()}")  # reversed!




#RequireGrad



# 2. The computation graph — peek at it
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = y + 2 * x + 1

print(f"z = {z.item()}")
print(f"z was created by: {z.grad_fn}")           # AddBackward
print(f"  which came from: {z.grad_fn.next_functions}")
print(f"y was created by: {y.grad_fn}")            # PowBackward
print(f"\n☝️ PyTorch recorded every operation!")
print("When you call z.backward(), it walks this chain in reverse.")




# 3. Gradient accumulation — the gotcha



x = torch.tensor(3.0, requires_grad=True)

y1 = x ** 2
y1.backward()
print(f"After first backward: x.grad = {x.grad}")   # 6.0

# x.grad.zero_() -> add this line to not accummulate the gradients
y2 = x ** 2
y2.backward()
print(f"After second backward: x.grad = {x.grad}")  # 12.0! Accumulated!

# This is why we call zero_grad()
x.grad.zero_()
y3 = x ** 2
y3.backward()
print(f"After zero + backward: x.grad = {x.grad}")  # 6.0 — fresh!


#CPU vs GPU

sizes = [100, 500, 1000, 2000, 4000]
cpu_times = []
gpu_times = []

for size in sizes:
    A_cpu = torch.randn(size, size)
    B_cpu = torch.randn(size, size)

    # CPU timing
    start = time.time()
    for _ in range(5):
        _ = A_cpu @ B_cpu
    cpu_time = (time.time() - start) / 5

    if torch.cuda.is_available():
        A_gpu = A_cpu.to('cuda')
        B_gpu = B_cpu.to('cuda')

        # Warmup
        _ = A_gpu @ B_gpu
        torch.cuda.synchronize()

        # GPU timing
        start = time.time()
        for _ in range(5):
            _ = A_gpu @ B_gpu
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 5
    else:
        gpu_time = float('nan')

    cpu_times.append(cpu_time * 1000)
    gpu_times.append(gpu_time * 1000)

    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"Size {size:>5}×{size:<5} | CPU: {cpu_time*1000:>8.2f}ms | GPU: {gpu_time*1000:>8.2f}ms | Speedup: {speedup:>6.1f}x")

# Plot it
if torch.cuda.is_available():
    fig, ax = plt.subplots(figsize=(10, 5))
    x_pos = range(len(sizes))
    ax.bar([i - 0.2 for i in x_pos], cpu_times, 0.35, label='CPU', color='#3b82f6')
    ax.bar([i + 0.2 for i in x_pos], gpu_times, 0.35, label='GPU', color='#ef4444')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([f"{s}×{s}" for s in sizes])
    ax.set_ylabel('Time (ms)')
    ax.set_title('Matrix Multiplication: CPU vs GPU', fontweight='bold', fontsize=14)
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()
    print("\n☝️ This is why NVIDIA is worth trillions.")




# Basic math — element-wise




a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("Element-wise operations:")
print(f"  a + b = {a + b}")
print(f"  a * b = {a * b}")      # element-wise multiply
print(f"  a @ b = {a @ b}")      # dot product!
print()

# Broadcasting in action
matrix = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
bias = torch.tensor([10.0, 20.0, 30.0])

result = matrix + bias  # bias gets added to EVERY row
print("Broadcasting:")
print(f"  Matrix shape: {matrix.shape}")
print(f"  Bias shape:   {bias.shape}")
print(f"  Result:\n{result}")
print("  ☝️ Bias was added to both rows automatically!")
print()

# Useful operations
x = torch.randn(3, 4)
print(f"Random tensor:\n{x}\n")
print(f"  Mean:    {x.mean():.4f}")
print(f"  Std:     {x.std():.4f}")
print(f"  Max:     {x.max():.4f}")
print(f"  Argmax:  {x.argmax()} (index of largest element)")
print(f"  Softmax: {torch.softmax(x[0], dim=0)}")
print(f"  ☝️ Softmax turns values into probabilities (sums to 1)")
