# Simple: y = x² + 2x + 1, find dy/dx at x=3
x = torch.tensor(3.0, requires_grad=True)  # ← "track this!"

y = x**2 + 2*x + 1  # y = 9 + 6 + 1 = 16
print(f"x = {x.item()}")
print(f"y = x² + 2x + 1 = {y.item()}")

y.backward()  # ← compute ALL gradients

print(f"dy/dx = {x.grad.item()}")
print(f"Check: dy/dx = 2x + 2 = 2(3) + 2 = {2*3 + 2} ✓")
print()
print("🤯 PyTorch computed the derivative automatically!")






x = torch.tensor(2.0, requires_grad=True)

# A more complex function
y = torch.sin(x) * torch.exp(x) + x**3

y.backward()

print(f"f(x) = sin(x) × eˣ + x³")
print(f"f(2) = {y.item():.4f}")
print(f"f'(2) = {x.grad.item():.4f}")
print()

# Verify numerically (finite difference)
h = 1e-5
x_val = 2.0
f_x = np.sin(x_val) * np.exp(x_val) + x_val**3
f_x_h = np.sin(x_val + h) * np.exp(x_val + h) + (x_val + h)**3
numerical_grad = (f_x_h - f_x) / h
print(f"Numerical check: {numerical_grad:.4f}")
print("☝️ They match! Autograd is correct no matter how complex the function.")





# This is what actually happens during training
W = torch.randn(3, 2, requires_grad=True)  # a weight matrix
x = torch.tensor([1.0, 0.5])                # input
target = torch.tensor([1.0, 0.0, 0.0])      # target output

# Forward pass
prediction = W @ x                    # matrix × vector
loss = ((prediction - target)**2).sum()  # MSE loss

print(f"Weight matrix W:\n{W.data}")
print(f"Input: {x}")
print(f"Prediction: {prediction.data}")
print(f"Target: {target}")
print(f"Loss: {loss.item():.4f}")
print()

# Backward pass — gradients for EVERY element of W
loss.backward()

print(f"Gradients for W (∂Loss/∂W):\n{W.grad}")
print()
print("☝️ PyTorch computed the gradient for all 6 weights at once!")
print("   This is what happens billions of times during training.")





print("=" * 55)
print("CLASS 2 (MANUAL) vs CLASS 4 (AUTOGRAD)")
print("=" * 55)
print()

# The problem: y = wx + b, loss = (y - target)²
# Manual version (what we did in Class 2)
print("--- Manual (Class 2 style) ---")
w_manual = 0.5
b_manual = 0.1
x_val = 2.0
target_val = 3.0
lr = 0.01

y_manual = w_manual * x_val + b_manual
loss_manual = (y_manual - target_val) ** 2

# Hand-computed gradients
dL_dy = 2 * (y_manual - target_val)
dy_dw = x_val
dy_db = 1.0
dL_dw = dL_dy * dy_dw
dL_db = dL_dy * dy_db

w_manual -= lr * dL_dw
b_manual -= lr * dL_db

print(f"  Loss: {loss_manual:.4f}")
print(f"  dL/dw = {dL_dw:.4f}")
print(f"  dL/db = {dL_db:.4f}")
print(f"  Updated w: {w_manual:.4f}")
print(f"  Updated b: {b_manual:.4f}")
print()

# PyTorch autograd version
print("--- Autograd (Class 4 style) ---")
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
x_t = torch.tensor(2.0)
target_t = torch.tensor(3.0)

y = w * x_t + b
loss = (y - target_t) ** 2

loss.backward()  # ← ONE LINE does all the work

with torch.no_grad():
    w -= lr * w.grad
    b -= lr * b.grad

print(f"  Loss: {loss.item():.4f}")
print(f"  dL/dw = {w.grad.item():.4f}")
print(f"  dL/db = {b.grad.item():.4f}")
print(f"  Updated w: {w.item():.4f}")
print(f"  Updated b: {b.item():.4f}")
print()

print("✅ Same answers. But autograd scales to millions of parameters.")
