import torch.nn as nn
from .attention import MultiHeadAttention
from .utils import SublayerConnection, PositionwiseFeedForward

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden_units=256, num_heads=4, d_ff=1024, dropout=0.1):
        """
        :param hidden_units: hidden_units size of transformer
        :param num_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_units
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, hidden_units=hidden_units, dropout=dropout)
        self.attention_sublayer = SublayerConnection(hidden_units=hidden_units, dropout=dropout)
        self.pwff = PositionwiseFeedForward(hidden_units=hidden_units, d_ff=d_ff, dropout=dropout)
        self.pwff_sublayer = SublayerConnection(hidden_units=hidden_units, dropout=dropout)
        self.dropoutlayer = nn.Dropout(p=dropout)

    def forward(self, input_enc, mask=None):
        # we need dynamic mask for the attention forward (sublayer module also has parameters, namely layernorm)
        # x: (b x L x dim_model)
        # mask: (b x L x L), set False to ignore that point
        x = self.attention_sublayer(lambda _x: self.attention(_x, mask=mask), input_enc)
        x = self.pwff_sublayer(self.pwff, x)
        return self.dropoutlayer(x)