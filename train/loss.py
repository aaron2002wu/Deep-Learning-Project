# losses.py
import torch
import torch.nn as nn

class ResidualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, model_accel, residual_pred, measured_accel):
        # model_accel: from Fossen 3DOF model
        # residual_pred: from neural network
        # measured_accel: from sensor or ground truth

        predicted_accel = model_accel + residual_pred
        loss = self.mse(predicted_accel, measured_accel)
        return loss
