from .feed_forward import FeedForwardLayer
from .abstract_block import AbstractBlock
from .mha import MultiHeadAttention
from .layer_norm import LayerNorm
from .dropout import Dropout

class TransformerEncoderLayer(AbstractBlock):
    """
    TransformerEncoderLayer(x) = 
    LN(x + Dropout(SelfAttention(x))) →
    LN(x + Dropout(FFN(x)))
    """

    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        use_rope: bool = False,
        max_seq_length: int = 512, 
        dropout: float = 0.1
    ):
        # Initialize needed components
        self.self_attn = MultiHeadAttention(d_model, num_heads, use_rope, max_seq_length)
        self.feed_forward = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = Dropout(p = dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = Dropout(p = dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(Q = x, K = x, V = x, mask = mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

    def parameters(self):
        return (
            self.self_attn.parameters() +
            self.feed_forward.parameters() +
            self.norm1.parameters() +
            self.norm2.parameters()
        )

    def backward(self, dLdy):
        # Backprop through 2nd LayerNorm
        d_res2 = self.norm2.backward(dLdy)

        # FFN branch
        d_ff_out = self.dropout2.backward(d_res2)
        d_x1_from_ff = self.feed_forward.backward(d_ff_out)

        # Residual merge at FFN block
        d_x1 = d_res2 + d_x1_from_ff

        # Backprop through 1st LayerNor
        d_res1 = self.norm1.backward(d_x1)

        # Attention branch
        d_attn_out = self.dropout1.backward(d_res1)
        d_x0_from_attn = self.self_attn.backward(d_attn_out)

        return d_res1 + d_x0_from_attn