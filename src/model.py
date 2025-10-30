import torch
import torch.nn as nn
from config import * 

class StackedLSTM(nn.Module):
    def __init__(self, input_dim=N_FEATURES, hidden_dim=N_HIDDENS, num_layers=N_LAYERS, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # === 1. LSTM Encoder ===
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout
        )

        # === 2. FC Decoder ===
        self.fc = nn.Linear(hidden_dim * 2, input_dim)
        self.relu = nn.LeakyReLU(0.1)

        # === 3. Feature Attention Layers ===
        self.dense1 = nn.Linear(input_dim, input_dim // 2)
        self.dense2 = nn.Linear(input_dim // 2, input_dim)
        self.sigmoid = nn.Sigmoid()

        # === 4. Learnable mix factor (skip connection) ===
        self.w = nn.Parameter(torch.FloatTensor([-0.01]), requires_grad=True)

    def forward(self, x):
        # x: (batch, window_size, n_features)
        B, T, F = x.size()

        # === Feature Attention ===
        pool = nn.AdaptiveAvgPool1d(1)
        attention_x = x.transpose(1, 2)         # (B, F, T)
        attention = pool(attention_x)           # (B, F, 1)
        connection = attention.view(B, F)       # (B, F)

        # Dense layers → attention weights (0~1)
        attention = self.relu(attention.squeeze(-1))
        attention = self.relu(self.dense1(attention))
        attention = self.sigmoid(self.dense2(attention))  # (B, F)

        # === LSTM Encoder ===
        x = x.transpose(0, 1)                   # (T, B, F)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(self.relu(outs[-1]))      # (B, F)

        # === Mix-up Skip Connection ===
        mix_factor = self.sigmoid(self.w)
        output = mix_factor * connection * attention + out * (1 - mix_factor)
        return output
