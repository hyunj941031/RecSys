import torch.nn as nn
from .layer_norm import LayerNorm

class SublayerConnection(nn.Module):
    # layer가 많아지면 학습이 잘 안되는 현상
    # 따라서, sub-layer들에 각각 dropout을 한 후 서로 residual connection을 적용하고, layer normalization 적용
    def __init__(self, hidden_units, dropout):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout = dropout

        self.norm = LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sublayer, x):
        "Apply residual connection to any sublayer with the same size."
        r = self.norm(x)
        r = sublayer(r)

        r = self.dropout(r)

        return x + r