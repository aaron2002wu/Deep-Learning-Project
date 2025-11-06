# train_lstm.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from models.lstm import ResidualLSTM
from models.physics import Fossen3DOF

# Hyperparameters
in_dim = 5      # [u, v, r, thrust_L, thrust_R]
out_dim = 3     # [u_dot, v_dot, r_dot]
hidden_dim = 128
num_layers = 2
seq_len = 10    # Sequence length for LSTM
lr = 1e-3
epochs = 100
batch_size = 64
dropout = 0.1

# Data Loading
csv_path = "~/Downloads/processed_new.csv"
df = pd.read_csv(csv_path, parse_dates=["time"])
print(df.columns)

inputs_cols = ["u_filt", "v_filt", "r_filt", "cmd_thrust.port", "cmd_thrust.starboard"]
target_cols = ["du_dt", "dv_dt", "dr_dt"]

print(df[inputs_cols + target_cols].isna().sum())
print(df[inputs_cols + target_cols].describe())

inputs = torch.tensor(df[inputs_cols].values, dtype=torch.float32)
measured_accel = torch.tensor(df[target_cols].values, dtype=torch.float32)

print("Inputs shape:", inputs.shape)            # (N, 5)
print("Measured accel shape:", measured_accel.shape)  # (N, 3)

# Create sequences for LSTM
class SequenceDataset(Dataset):
    def __init__(self, inputs, targets, seq_len):
        self.inputs = inputs
        self.targets = targets
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.inputs) - self.seq_len + 1
    
    def __getitem__(self, idx):
        # Get sequence of inputs: (seq_len, in_dim)
        x_seq = self.inputs[idx:idx+self.seq_len]
        # Get target for the last timestep: (out_dim,)
        y = self.targets[idx+self.seq_len-1]
        return x_seq, y

dataset = SequenceDataset(inputs, measured_accel, seq_len)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(loader)}")

# Model + Loss + Optimizer
model = ResidualLSTM(
    in_dim=in_dim, 
    hidden_dim=hidden_dim, 
    num_layers=num_layers, 
    out_dim=out_dim,
    dropout=dropout
)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

fossen_model = Fossen3DOF()

loss_history = []

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_seq_batch, measured_accel_batch in loader:
        # x_seq_batch shape: (batch, seq_len, in_dim)
        # measured_accel_batch shape: (batch, out_dim)
        
        # Get the last timestep values for physics model
        # x_seq_batch[:, -1, :] shape: (batch, in_dim)
        u, v, r, tL, tR = x_seq_batch[:, -1, :].T
        
        # Step 1: Physics-based prediction
        model_accel = fossen_model.forward(u, v, r, tL, tR)
        
        # Step 2: Compute residual target
        residual_target = measured_accel_batch - model_accel
        
        # Step 3: Neural network predicts residual
        residual_pred = model(x_seq_batch)  # (batch, out_dim)
        
        # Step 4: Compute loss
        loss = criterion(residual_pred, residual_target)
        
        # Step 5: Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)
    
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

# Plot training loss curve
plt.figure()
plt.plot(loss_history)
plt.title("LSTM Training Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("lstm_training_loss.png")
print("✅ Training loss plot saved as lstm_training_loss.png")

# Save the trained model
torch.save(model.state_dict(), "residual_lstm.pth")
print("✅ Model saved as residual_lstm.pth")

# -------------------- EVALUATION --------------------
model.eval()
with torch.no_grad():
    # Create sequences for evaluation
    eval_dataset = SequenceDataset(inputs, measured_accel, seq_len)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    all_residual_preds = []
    all_measured_accels = []
    all_model_accels = []
    
    for x_seq_batch, measured_accel_batch in eval_loader:
        u, v, r, tL, tR = x_seq_batch[:, -1, :].T
        model_accel = fossen_model.forward(u, v, r, tL, tR)
        residual_pred = model(x_seq_batch)
        
        all_residual_preds.append(residual_pred)
        all_measured_accels.append(measured_accel_batch)
        all_model_accels.append(model_accel)
    
    residual_pred = torch.cat(all_residual_preds, dim=0)
    measured_accel_eval = torch.cat(all_measured_accels, dim=0)
    model_accel = torch.cat(all_model_accels, dim=0)
    hybrid_accel = model_accel + residual_pred
    
    # Pad the beginning for plotting (since we lose seq_len-1 samples)
    pad_size = seq_len - 1
    residual_pred_padded = torch.cat([
        torch.zeros(pad_size, out_dim),
        residual_pred
    ], dim=0)
    hybrid_accel_padded = torch.cat([
        torch.zeros(pad_size, out_dim),
        hybrid_accel
    ], dim=0)
    model_accel_padded = torch.cat([
        torch.zeros(pad_size, out_dim),
        model_accel
    ], dim=0)
    measured_accel_padded = measured_accel[:pad_size]
    measured_accel_padded = torch.cat([
        measured_accel_padded,
        measured_accel_eval
    ], dim=0)

# -------------------- QUANTITATIVE EVALUATION --------------------
# Compute per-axis MSE and MAE (on evaluation data)
mse_hybrid = torch.mean((hybrid_accel - measured_accel_eval)**2, dim=0)
mae_hybrid = torch.mean(torch.abs(hybrid_accel - measured_accel_eval), dim=0)

mse_model = torch.mean((model_accel - measured_accel_eval)**2, dim=0)
mae_model = torch.mean(torch.abs(model_accel - measured_accel_eval), dim=0)

acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]
print("\n--- Evaluation Metrics ---")
for i in range(3):
    print(f"{acc_labels[i]}: MSE hybrid={mse_hybrid[i]:.6f}, MAE hybrid={mae_hybrid[i]:.6f} | "
          f"MSE physics={mse_model[i]:.6f}, MAE physics={mae_model[i]:.6f}")

# -------------------- PLOTTING: Model, Residual, Hybrid, Measured --------------------
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

acc_labels = ["u_dot (surge)", "v_dot (sway)", "r_dot (yaw rate)"]
colors = ["blue", "green", "red", "black"]
linestyles = ["--", ":", "-.", "-"]  # model, residual, hybrid, measured

for i in range(3):
    axs[i].plot(model_accel_padded[:, i].numpy(), label=f"Physics {acc_labels[i]}", 
                color=colors[0], linestyle=linestyles[0], alpha=0.7)
    axs[i].plot(residual_pred_padded[:, i].numpy(), label=f"Residual {acc_labels[i]}", 
                color=colors[1], linestyle=linestyles[1], alpha=0.7)
    axs[i].plot(hybrid_accel_padded[:, i].numpy(), label=f"Hybrid {acc_labels[i]}", 
                color=colors[2], linestyle=linestyles[2], alpha=0.7)
    axs[i].plot(measured_accel_padded[:, i].numpy(), label=f"Measured {acc_labels[i]}", 
                color=colors[3], linestyle=linestyles[3], alpha=0.7)
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylabel("Acceleration [m/s²]")

axs[2].set_xlabel("Sample index")
plt.suptitle("LSTM: Measured vs Physics vs Residual vs Hybrid Accelerations")
plt.tight_layout()
plt.savefig("lstm_results.png")
print("✅ Results plot saved as lstm_results.png")
plt.show()