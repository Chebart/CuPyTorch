from .rope import RotaryPositionalEmbeddings
from .abstract_block import AbstractBlock
from .softmax import Softmax
from .linear import Linear

class MultiHeadAttention(AbstractBlock):
    """MultiHeadAttention(Q, K, V) = Concat(head_1, …, head_h) * W_O"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: int = 4096,
    ):
        # check that model dim can be split among heads
        assert d_model % num_heads == 0

        # Initialize needed vars
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # RoPE flag
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.d_k,
                max_seq_len=max_seq_len,
            )
        else:
            self.rope = None

        # Initialize needed components
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        self.softmax = Softmax()

    def split_heads(self, x):
        # Split last dim into (num_heads, head_dim)
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, x):
        # Combine heads back
        B, H, T, D = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Attention score computation and normalization
        scores = Q @ K.transpose(0, 1, 3, 2) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Get attention probabilities
        probs = self.softmax(scores, dim=-1)

        # Cache tensors
        self.attn_probs = probs
        self.Q = Q
        self.K = K
        self.V = V

        return probs @ V

    def forward(self, Q, K, V, mask=None):
        # Get linear projections
        self.Q_lin = self.W_q(Q)
        self.K_lin = self.W_k(K)
        self.V_lin = self.W_v(V)

        # Split projections on head
        Qh = self.split_heads(self.Q_lin)
        Kh = self.split_heads(self.K_lin)
        Vh = self.split_heads(self.V_lin)

        # Apply rotary positional embeddings to Q and K
        if self.use_rope:
            # (B, H, T, D) -> (B, T, H, D)
            Qh = Qh.transpose(0, 2, 1, 3)
            Kh = Kh.transpose(0, 2, 1, 3)

            Qh = self.rope(Qh)
            Kh = self.rope(Kh)

            # (B, T, H, D) -> (B, H, T, D)
            Qh = Qh.transpose(0, 2, 1, 3)
            Kh = Kh.transpose(0, 2, 1, 3)

        # Calculate scaled dot-product attention
        attn_out = self.scaled_dot_product_attention(Qh, Kh, Vh, mask)

        return self.W_o(self.combine_heads(attn_out))

    def parameters(self):
        return (
            self.W_q.parameters() +
            self.W_k.parameters() +
            self.W_v.parameters() +
            self.W_o.parameters()
        )

    def backward(self, dLdy):
        # Output projection backward
        d_out = self.W_o.backward(dLdy)
        d_attn = self.split_heads(d_out)

        # Backprop through attention
        d_probs = d_attn @ self.V.transpose(0, 1, 3, 2)
        dV = self.attn_probs.transpose(0, 1, 3, 2) @ d_attn

        # Backprop through score normalization
        d_scores = self.softmax.backward(d_probs)
        d_scores /= (self.d_k ** 0.5)

        # Backprop to Q and K
        dQ = d_scores @ self.K
        dK = d_scores.transpose(0, 1, 3, 2) @ self.Q

        # Calculate RoPE grad
        if self.use_rope:
            # (B, H, T, D) -> (B, T, H, D)
            dQ = dQ.transpose(0, 2, 1, 3)
            dK = dK.transpose(0, 2, 1, 3)

            dQ = self.rope.backward(dQ)
            dK = self.rope.backward(dK)

            # (B, T, H, D) -> (B, H, T, D)
            dQ = dQ.transpose(0, 2, 1, 3)
            dK = dK.transpose(0, 2, 1, 3)

        # Merge heads and backprop through linear layers
        dQ = self.W_q.backward(self.combine_heads(dQ))
        dK = self.W_k.backward(self.combine_heads(dK))
        dV = self.W_v.backward(self.combine_heads(dV))

        return dQ + dK + dV