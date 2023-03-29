import torch.nn as nn
import torch.nn.functional as F
import torch
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate) # dropout rate

    def forward(self, Q, K, V, mask=None, dropout=None):
        """
            b = batch, ? = num_heads, L = sequence_len

            Q: (b x ? x L x dim_Q)
            K: (b x ? x L x dim_K)
            V: (b x ? x L x dim_V)
            ?: 1 (squeezed) or h (multi-head)
            mask: (b x ? x L x L)
            dropout: nn.Module
            assuming dim_Q = dim_K
        """

        # A: (b x ? x L x L)
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units)

        # apply mask (the logit value of a padding token should be minus infinity)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9) # 유사도가 0인 지점은 -infinity로 보내서 softmax 결과가 0이 아니게 함
        
        # getting normalized(probability) weights through softmax (when padding token, it'll be 0)
        # P: (b x ? x L x L)
        p_attn = F.softmax(attn_score, dim=-1)
        
        # apply dropout (with given dropout)
        if dropout is not None:
            p_attn = self.dropout(p_attn) # attention distribution 상대적 중요도

        # (b x ? x L x L) @ (b x ? x L x dim_V) -> (b x ? x L x dim_V)
        output = torch.matmul(p_attn, V)

        return output, p_attn