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
from models.mlp import ResidualMLP, MLP
from models.physics import Fossen3DOF  # physics-based model

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
in_dim = 5      # [u, v, r, thrust_port, thrust_stbd]
out_dim = 3     # [u_dot, v_dot, r_dot]
hidden = 128
lr = 1e-3
epochs = 100
batch_size = 64

# -----------------------------
# DATA LOADING
# -----------------------------
csv_path = os.path.expanduser("~/Downloads/exp2/processed_forces.csv")
df = pd.read_csv(csv_path, parse_dates=["time"])
print("Columns in CSV:", df.columns.tolist())
print(f"Loaded {len(df)} rows")

inputs_cols = ["u_filt", "v_filt", "r_filt", "cmd_thrust.port", "cmd_thrust.stbd"]
target_cols = ["du_dt", "dv_dt", "dr_dt"]

missing_cols = [c for c in inputs_cols + target_cols if c not in df.columns]
if missing_cols:
    print(f"ERROR: Missing columns: {missing_cols}")
    sys.exit(1)

print("\nNaN counts:")
print(df[inputs_cols + target_cols].isna().sum())

df_clean = df[inputs_cols + target_cols].dropna()
print(f"\nRows after dropping NaN: {len(df_clean)}")

print("\nData statistics:")
print(df_clean.describe())

inputs = torch.tensor(df_clean[inputs_cols].values, dtype=torch.float32)
measured_accel = torch.tensor(df_clean[target_cols].values, dtype=torch.float32)

print("\nTensor shapes:")
print("Inputs:", inputs.shape)
print("Measured accel:", measured_accel.shape)

dataset = TensorDataset(inputs, measured_accel)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -----------------------------
# RESIDUAL MODEL + PHYSICS
# -----------------------------
residual_model = ResidualMLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(residual_model.parameters(), lr=lr)
fossen_model = Fossen3DOF()

# -----------------------------
# DRY RUN
# -----------------------------
print("\n🔍 Dry run...")
x_batch, measured_accel_batch = next(iter(loader))
u, v, r, tPort, tStbd = x_batch.T
model_accel = fossen_model.forward(u, v, r, tPort, tStbd)
residual_target = measured_accel_batch - model_accel
residual_pred = residual_model(x_batch)
print("Residual pred:", residual_pred.shape)
print("Initial loss:", criterion(residual_pred, residual_target).item())
print("Dry run passed.\n")

# -----------------------------
# TRAIN RESIDUAL MLP
# -----------------------------
loss_history = []

print("\n========================")
print("TRAINING RESIDUAL MODEL")
print("========================")

for epoch in range(epochs):
    total_loss = 0
    for x_batch, measured_accel_batch in loader:
        u, v, r, tPort, tStbd = x_batch.T
        model_accel = fossen_model.forward(u, v, r, tPort, tStbd)
        residual_target = measured_accel_batch - model_accel
        residual_pred = residual_model(x_batch)

        loss = criterion(residual_pred, residual_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / len(loader)
    loss_history.append(avg)
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} Loss: {avg:.6f}")

plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title("Residual MLP Training Loss")
plt.savefig("training_loss_residual.png")
print("Saved residual model loss plot.")

torch.save(residual_model.state_dict(), "residual_mlp.pth")
print("Saved residual_mlp.pth")

# -----------------------------
# EVALUATE HYBRID MODEL
# -----------------------------
residual_model.eval()
with torch.no_grad():
    u, v, r, tPort, tStbd = inputs.T
    model_accel = fossen_model.forward(u, v, r, tPort, tStbd)
    residual_pred = residual_model(inputs)
    hybrid_accel = model_accel + residual_pred

# Metrics
mse_hybrid = torch.mean((hybrid_accel - measured_accel)**2, dim=0)
mae_hybrid = torch.mean(torch.abs(hybrid_accel - measured_accel), dim=0)
mse_model = torch.mean((model_accel - measured_accel)**2, dim=0)
mae_model = torch.mean(torch.abs(model_accel - measured_accel), dim=0)

# -----------------------------
# TRAIN PURE MLP BASELINE
# -----------------------------
print("\n========================")
print("TRAINING PURE MLP MODEL")
print("========================")

mlp_model = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)
optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=lr)

mlp_loss_hist = []
for epoch in range(epochs):
    total = 0
    for x_batch, y_batch in loader:
        pred = mlp_model(x_batch)
        loss = criterion(pred, y_batch)
        optimizer_mlp.zero_grad()
        loss.backward()
        optimizer_mlp.step()
        total += loss.item()
    avg = total / len(loader)
    mlp_loss_hist.append(avg)
    if (epoch+1) % 10 == 0:
        print(f"[MLP] Epoch {epoch+1}/{epochs} Loss: {avg:.6f}")

torch.save(mlp_model.state_dict(), "pure_mlp.pth")
print("Saved pure_mlp.pth")

mlp_model.eval()
with torch.no_grad():
    mlp_pred = mlp_model(inputs)

mse_mlp = torch.mean((mlp_pred - measured_accel)**2, dim=0)
mae_mlp = torch.mean(torch.abs(mlp_pred - measured_accel), dim=0)

# -----------------------------
# PRINT METRICS
# -----------------------------
print("\n========================================")
print("            EVALUATION RESULTS")
print("========================================")
labels = ["u_dot", "v_dot", "r_dot"]

for i in range(3):
    print(f"\n{labels[i]}:")
    print(f" Physics: MSE={mse_model[i]:.6f}  MAE={mae_model[i]:.6f}")
    print(f" Hybrid : MSE={mse_hybrid[i]:.6f}  MAE={mae_hybrid[i]:.6f}")
    print(f" PureMLP: MSE={mse_mlp[i]:.6f}  MAE={mae_mlp[i]:.6f}")

# -----------------------------
# PLOTS
# -----------------------------
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
time_vals = range(len(measured_accel))

for i in range(3):
    axs[i].plot(time_vals, measured_accel[:, i], label="Measured", color="black")
    axs[i].plot(time_vals, model_accel[:, i], label="Physics", linestyle="--")
    axs[i].plot(time_vals, hybrid_accel[:, i], label="Hybrid", linestyle="-.")
    axs[i].plot(time_vals, mlp_pred[:, i], label="PureMLP", linestyle=":")
    axs[i].set_ylabel(labels[i])
    axs[i].grid(True)
    axs[i].legend()

axs[2].set_xlabel("Samples")
plt.suptitle("Acceleration Comparison: Physics vs Hybrid vs PureMLP")
plt.tight_layout()
plt.savefig("accel_compare_all.png")
plt.show()

print("\n========================")
print("TRAINING COMPLETE")
print("========================")
