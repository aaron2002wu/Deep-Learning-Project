# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import ResidualMLP
from losses import ResidualLoss
from fossen_model import fossen_3dof_accel  # physics-based model function

# --------------------
# Hyperparameters
# --------------------
in_dim = 5      # [u, v, r, thrust_L, thrust_R]
out_dim = 3     # [u_dot, v_dot, r_dot]
hidden = 128
lr = 1e-3
epochs = 100

# --------------------
# Example Data Loading
# --------------------
# Assume you already processed your CSV to tensors
u = torch.randn(1000, 1)
v = torch.randn(1000, 1)
r = torch.randn(1000, 1)
thrust_L = torch.randn(1000, 1)
thrust_R = torch.randn(1000, 1)
measured_accel = torch.randn(1000, 3)

inputs = torch.cat([u, v, r, thrust_L, thrust_R], dim=1)
dataset = TensorDataset(inputs, measured_accel)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# --------------------
# Model + Loss + Opt
# --------------------
model = ResidualMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
criterion = ResidualLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --------------------
# Training Loop
# --------------------
for epoch in range(epochs):
    total_loss = 0
    for x_batch, measured_accel in loader:
        # Step 1: Compute model-based accel (Fossen 3DOF)
        u, v, r, tL, tR = x_batch[:,0], x_batch[:,1], x_batch[:,2], x_batch[:,3], x_batch[:,4]
        model_accel = fossen_3dof_accel(u, v, r, tL, tR)

        # Step 2: Neural network residual prediction
        residual_pred = model(x_batch)

        # Step 3: Compute loss
        loss = criterion(model_accel, residual_pred, measured_accel)

        # Step 4: Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.6f}")
