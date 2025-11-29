# train.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
# Import models
from models.mlp import ResidualMLP
from models.physics import Fossen3DOF  # physics-based model

# Hyperparameters
in_dim = 5      # [u, v, r, thrust_port, thrust_stbd]
out_dim = 3     # [u_dot, v_dot, r_dot]
hidden = 128
lr = 1e-3
epochs = 100
batch_size = 64

# Data Loading
csv_path = os.path.expanduser("~/Downloads/exp2/processed_forces.csv")
df = pd.read_csv(csv_path, parse_dates=["time"])
print("Columns in CSV:", df.columns.tolist())
print(f"Loaded {len(df)} rows")

# Updated column names based on your new CSV
inputs_cols = ["u_filt", "v_filt", "r_filt", "cmd_thrust.port", "cmd_thrust.stbd"]
target_cols = ["du_dt", "dv_dt", "dr_dt"]  # calculated accelerations from finite differences

# Check for missing columns
missing_cols = [col for col in inputs_cols + target_cols if col not in df.columns]
if missing_cols:
    print(f"ERROR: Missing columns: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)

# Check for NaN values
print("\nNaN counts:")
print(df[inputs_cols + target_cols].isna().sum())

# Drop rows with NaN values
df_clean = df[inputs_cols + target_cols].dropna()
print(f"\nRows after dropping NaN: {len(df_clean)} (dropped {len(df) - len(df_clean)})")

# Statistics
print("\nData statistics:")
print(df_clean.describe())

# Convert to tensors
inputs = torch.tensor(df_clean[inputs_cols].values, dtype=torch.float32)
measured_accel = torch.tensor(df_clean[target_cols].values, dtype=torch.float32)

print("\nTensor shapes:")
print("Inputs shape:", inputs.shape)            # (N, 5)
print("Measured accel shape:", measured_accel.shape)  # (N, 3)

# Check for any remaining NaN or inf values
if torch.isnan(inputs).any() or torch.isinf(inputs).any():
    print("WARNING: NaN or Inf values in inputs!")
if torch.isnan(measured_accel).any() or torch.isinf(measured_accel).any():
    print("WARNING: NaN or Inf values in measured_accel!")

# Create dataset and dataloader
dataset = TensorDataset(inputs, measured_accel)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model + Loss + Optimizer
model = ResidualMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Physics-based model
fossen_model = Fossen3DOF()

# Dry run check
print("\n🔍 Running dry run (single batch check)...")
x_batch, measured_accel_batch = next(iter(loader))
print("x_batch:", x_batch.shape)
print("measured_accel_batch:", measured_accel_batch.shape)

# Unpack - note: columns are now port and stbd instead of L and R
u, v, r, tPort, tStbd = x_batch.T
print("Sample thrust values - Port:", tPort[:5], "Stbd:", tStbd[:5])

# Physics-based prediction
model_accel = fossen_model.forward(u, v, r, tPort, tStbd)
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

loss_history = []

# Training Loop
for epoch in range(epochs):
    total_loss = 0
    for x_batch, measured_accel_batch in loader:
        # Unpack with correct names
        u, v, r, tPort, tStbd = x_batch.T
        
        # Step 1: Physics-based prediction
        model_accel = fossen_model.forward(u, v, r, tPort, tStbd)

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
    
    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

# Plot training loss curve
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.title("Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss.png")
print("✅ Training loss plot saved as training_loss.png")

# Save the trained model
torch.save(model.state_dict(), "residual_mlp.pth")
print("✅ Model saved as residual_mlp.pth")

# -------------------- EVALUATION --------------------
model.eval()
with torch.no_grad():
    u, v, r, tPort, tStbd = inputs.T
    model_accel = fossen_model.forward(u, v, r, tPort, tStbd)
    residual_pred = model(inputs)
    hybrid_accel = model_accel + residual_pred

# -------------------- QUANTITATIVE EVALUATION --------------------
# Compute per-axis MSE and MAE
mse_hybrid = torch.mean((hybrid_accel - measured_accel)**2, dim=0)
mae_hybrid = torch.mean(torch.abs(hybrid_accel - measured_accel), dim=0)

mse_model = torch.mean((model_accel - measured_accel)**2, dim=0)
mae_model = torch.mean(torch.abs(model_accel - measured_accel), dim=0)

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]
for i in range(3):
    print(f"\n{acc_labels[i]}:")
    print(f"  Hybrid  - MSE: {mse_hybrid[i]:.6f}, MAE: {mae_hybrid[i]:.6f}")
    print(f"  Physics - MSE: {mse_model[i]:.6f}, MAE: {mae_model[i]:.6f}")
    improvement = ((mse_model[i] - mse_hybrid[i]) / mse_model[i] * 100).item()
    print(f"  Improvement: {improvement:.2f}%")

# -------------------- PLOTTING: Model, Residual, Hybrid, Measured --------------------
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]
colors = ["blue", "green", "red", "black"]
linestyles = ["--", ":", "-.", "-"]  # model, residual, hybrid, measured

# Use time index if available, otherwise sample index
if 'time' in df_clean.columns:
    time_vals = df_clean.index.values
    xlabel = "Sample index"
else:
    time_vals = range(len(measured_accel))
    xlabel = "Sample index"

for i in range(3):
    axs[i].plot(time_vals, model_accel[:, i], label=f"Physics {acc_labels[i]}", 
                color=colors[0], linestyle=linestyles[0], alpha=0.7)
    axs[i].plot(time_vals, residual_pred[:, i], label=f"Residual {acc_labels[i]}", 
                color=colors[1], linestyle=linestyles[1], alpha=0.7)
    axs[i].plot(time_vals, hybrid_accel[:, i], label=f"Hybrid {acc_labels[i]}", 
                color=colors[2], linestyle=linestyles[2], linewidth=2)
    axs[i].plot(time_vals, measured_accel[:, i], label=f"Measured {acc_labels[i]}", 
                color=colors[3], linestyle=linestyles[3], alpha=0.8)
    axs[i].legend(loc='upper right')
    axs[i].grid(True, alpha=0.3)
    axs[i].set_ylabel("Acceleration [m/s² or rad/s²]")

axs[2].set_xlabel(xlabel)
plt.suptitle("Measured vs Physics vs Residual vs Hybrid Accelerations")
plt.tight_layout()
plt.savefig("acceleration_comparison.png", dpi=150)
print("✅ Acceleration comparison plot saved as acceleration_comparison.png")

plt.show()

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)