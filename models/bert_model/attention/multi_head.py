import torch.nn as nn
from .scaleddot import ScaledDotProductAttention
from ..utils.layer_norm import LayerNorm

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout=0.1):

        """
            dim_V should be equal to hidden_units / num_heads
            we assume dim_Q = dim_K = dim_V
        """

        super().__init__()
        assert hidden_units % num_heads == 0

        self.hidden_units = hidden_units
        self.num_heads = num_heads

        self.dim_V = hidden_units // num_heads

        # query, key, value, output 생성을 위한 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)

        self.W_O = nn.Linear(hidden_units * num_heads, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.layerNorm = LayerNorm(hidden_units)

    def forward(self, enc, mask):

        residual = enc

        batch_size, seqlen = enc.size(0), enc.size(1)

        # 1) Do all the linear projections in a batch from dim_model, then split into (num_heads x dim_V)
        # [process]
        # (1) linear(W): (b x L x dim_model) -> (b x L x dim_model)
        # (2) view: (b x L x dim_model) -> (b x L x num_heads x dim_V)
        # (3) transpose: (b x L x num_heads x dim_V) -> (b x num_heads x L x dim_V)
        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units).transpose(1, 2)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units).transpose(1, 2)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units).transpose(1, 2)

        # 2) Apply attention to the projected vectors in the batch
        # note that attenion only cares about the last two dimensions
        # output: (b x num_heads x L x dim_V)
        # Head별로 각기 다른 attention이 가능하도록 각각 attention에 통과시킴
        output, attn_dist = self.attention(Q, K, V, mask=mask, dropout=self.dropout) # attn_dist : (batch_size, num_heads, sequence_len, sequence_len)

        # 3) "concat" those heads using view
        # [process]
        # (1) transpose: (b x num_heads x L x dim_V) -> (b x L x num_heads x dim_V)
        # (2) contiguous: reorder memory inside GPU (no dimension change)
        # (3) view: (b x L x num_heads x dim_V) -> (b x L x dim_model)
        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous() # (batch_size, sequence_len, num_heads, d_model) / contiguous() : 가변적 메모리 할당
        output = output.view(batch_size, seqlen, -1) # (batch_size, sequence_len, d_model * num_heads)

        # 4) apply the final linear
        # X: (b x L x dim_model)
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)

        return output