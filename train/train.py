# train.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
# Import models
from models.mlp import ResidualMLP
from models.physics import Fossen3DOF  # physics-based model

# Hyperparameters
in_dim = 5      # [u, v, r, thrust_L, thrust_R]
out_dim = 3     # [u_dot, v_dot, r_dot]
hidden = 128
lr = 1e-3
epochs = 100
batch_size = 64

# Replace with Data Loading
# u = torch.randn(1000, 1)
# v = torch.randn(1000, 1)
# r = torch.randn(1000, 1)
# thrust_L = torch.randn(1000, 1)
# thrust_R = torch.randn(1000, 1)
# measured_accel = torch.randn(1000, 3)


csv_path = "~/Downloads/processed_new.csv"
df = pd.read_csv(csv_path,parse_dates=["time"])
print(df.columns)


inputs_cols = ["u_filt", "v_filt", "r_filt", "cmd_thrust.port", "cmd_thrust.starboard"]
target_cols = ["du_dt", "dv_dt", "dr_dt"]  # measured accelerations; SHOULD WE USE IMU OR CALCULATED ACCELS?

print(df[inputs_cols + target_cols].isna().sum())
print(df[inputs_cols + target_cols].describe())

inputs = torch.tensor(df[inputs_cols].values, dtype=torch.float32)
measured_accel = torch.tensor(df[target_cols].values, dtype=torch.float32)


print("Inputs shape:", inputs.shape)            # (N, 5)
print("Measured accel shape:", measured_accel.shape)  # (N, 3)

dataset = TensorDataset(inputs, measured_accel)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model + Loss + Optimizer
model = ResidualMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# --------------------
print("\n🔍 Running dry run (single batch check)...")

x_batch, measured_accel_batch = next(iter(loader))
print("x_batch:", x_batch.shape)
print("measured_accel_batch:", measured_accel_batch.shape)

# Unpack
u, v, r, tL, tR = x_batch.T

# Physics-based prediction
model_accel = Fossen3DOF.forward(u, v, r, tL, tR)
print("model_accel:", model_accel.shape)

# Residual target
residual_target = measured_accel_batch - model_accel
print("residual_target:", residual_target.shape)

# NN prediction
residual_pred = model(x_batch)
print("residual_pred:", residual_pred.shape)

# Loss
loss = criterion(residual_pred, residual_target)
print("Initial loss:", loss.item())

print("✅ Dry run passed — proceeding to training.\n")


# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for x_batch, measured_accel_batch in loader:
        u, v, r, tL, tR = x_batch[:,0], x_batch[:,1], x_batch[:,2], x_batch[:,3], x_batch[:,4]

        # Step 1: Physics-based prediction
        model_accel = Fossen3DOF.forward(u, v, r, tL, tR)

        # Step 2: Compute residual target
        residual_target = measured_accel_batch - model_accel

        # Step 3: Neural network predicts residual
        residual_pred = model(x_batch)

        # Step 4: Compute loss
        loss = criterion(residual_pred, residual_target)

        # Step 5: Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.6f}")
