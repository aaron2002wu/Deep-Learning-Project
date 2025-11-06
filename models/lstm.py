import torch.nn as nn

class ResidualLSTM(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=128, num_layers=2, out_dim=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # LSTM output: (batch, seq_len, hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # Take the last timestep output
        # lstm_out[:, -1, :] shape: (batch, hidden_dim)
        output = self.fc(lstm_out[:, -1, :])
        
        return output  # (batch, out_dim)