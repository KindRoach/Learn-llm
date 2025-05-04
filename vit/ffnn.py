import torch.nn as nn


class FFNN(nn.Module):
    def __init__(self, d_emd, dropout=0.2):
        super().__init__()
        hidden_features = 4 * d_emd
        self.fc1 = nn.Linear(d_emd, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, d_emd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)  # [B, T, 4*d]
        x = self.act(x)
        x = self.fc2(x)  # [B, T, d]
        x = self.drop(x)
        return x
