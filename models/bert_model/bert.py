import numpy as np
import torch
import torch.nn as nn

from bert_model.transformer import TransformerBlock
from utils import fix_random_seed

class BERT4Rec(nn.Module):
    def __init__(self, num_users, num_items, sequence_len, device, num_layers=2, hidden_units=256, num_heads=4, dropout=0.1, random_seed=None):
        super().__init__()

        # params
        self.num_users = num_users
        self.num_items = num_items
        self.sequence_len = sequence_len
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.random_seed = random_seed
        self.device = device
        
        # set seed
        if random_seed is not None:
            fix_random_seed(random_seed)

        # 0: padding token
        # 1 ~ V: item tokens
        # V + 1: mask token

        # Embedding layers
        self.item_emb = nn.Embedding(num_items + 2, hidden_units, padding_idx=0) # padding : 0 / item : 1 ~ num_item + 1 /  mask : num_item + 2
        self.pos_emb = nn.Embedding(sequence_len, hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(dropout)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)

        # transformer layers
        self.transformers = nn.ModuleList([
            TransformerBlock(
                hidden_units=hidden_units,
                num_heads=num_heads,
                d_ff=hidden_units*4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_units, num_items + 1)
    
    def forward(self, tokens):
        L = tokens.shape[1]

        # mask for whether padding token or not in attention matrix (True if padding token)
        # [process] (b x L) -> (b x 1 x L) -> (b x L x L) -> (b x 1 x L x L)
        token_mask = torch.BoolTensor(tokens > 0).unsqueeze(1).repeat(1, L, 1).unsqueeze(1).to(self.device)

        # get embedding
        # [process] (b x L) -> (b x L x d)
        seqs = self.item_emb(torch.LongTensor(tokens).to(self.device))  # (batch_size, sequence_len, hidden_units)
        positions = np.tile(np.array(range(tokens.shape[1])), [tokens.shape[0], 1])  # (batch_size, sequence_len)
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))  # (batch_size, sequence_len, hidden_units)
        seqs = self.emb_layernorm(self.dropout(seqs))  # LayerNorm

        # apply multi-layered transformers
        # [process] (b x L x d) -> ... -> (b x L x d)
        for transformer in self.transformers:
            seqs = transformer(seqs, token_mask)

        # classifier
        # [process] (b x L x d) -> (b x L x (V + 1))
        out = self.out(seqs)

        return out