import torch.nn as nn
from .gelu import GELU

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, d_ff, dropout=0.1):
        super().__init__()

        self.w_1 = nn.Linear(hidden_units, d_ff)
        self.w_2 = nn.Linear(d_ff, hidden_units)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))