from .abstract_block import AbstractBlock
from ..data import Tensor

class RotaryPositionalEmbeddings(AbstractBlock):
    """RotaryPositionalEmbeddings(x, pos) = R(pos) * x"""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 10000.0,
    ):
        assert dim % 2 == 0, "RoPE dimension must be even"
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len

        # Compute inverse frequencies
        self.theta = 1.0 / (
            self.base ** (Tensor.arange(0, dim, 2, dtype="fp32") / dim)
        )
        # Build cos/sin cache
        self.build_rope_cache(max_seq_len)

    def build_rope_cache(
        self, 
        max_seq_len: int
    ):
        # (max_seq_len,)
        positions = Tensor.arange(max_seq_len, dtype = self.theta.dtype)
        # shape: (max_seq_len, dim//2)
        idx_theta = positions[:, None] * self.theta[None, :]
        # shape: (max_seq_len, dim//2, 2)
        self.cache = Tensor.stack([idx_theta.cos(), idx_theta.sin()], axis=-1)

    def forward(
        self, 
        x: Tensor
    ):
        # Get input dims
        batch, seq_len, n_heads, head_dim = x.shape

        # Check dims correction
        assert seq_len <= self.max_seq_len, f"RoPE seq_len mismatch: seq_len={seq_len}, max_seq_len={self.max_seq_len}"
        assert head_dim == self.dim, f"RoPE dim mismatch: head_dim={head_dim}, rope_dim={self.dim}"
        assert head_dim % 2 == 0, "RoPE requires even head_dim"

        # Select positions
        rope = self.cache[:seq_len]
        rope = rope.to(dtype = x.dtype, device = x.device)
        # (1, seq_len, 1, dim//2, 2)
        rope = rope[None, :, None, :, :]

        # (batch, seq_len, n_heads, dim//2, 2)
        x = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)

        # Apply rotation
        x1 = x[..., 0]
        x2 = x[..., 1]
        cos = rope[..., 0]
        sin = rope[..., 1]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin

        # (batch, seq_len, n_heads, dim//2, 2)
        out = Tensor.stack([out1, out2], axis=-1)

        return out.reshape(batch, seq_len, n_heads, head_dim)

    def parameters(self):
        return []

    def backward(self, dLdy: Tensor) -> Tensor:
        # Get grad dims
        batch, seq_len, n_heads, head_dim = dLdy.shape

        # Select positions
        rope = self.cache[:seq_len]
        rope = rope.to(dtype = dLdy.dtype, device = dLdy.device)
        rope = rope[None, :, None, :, :]

        dLdy = dLdy.reshape(batch, seq_len, n_heads, head_dim // 2, 2)

        # Apply inverse rotation
        dy1 = dLdy[..., 0]
        dy2 = dLdy[..., 1]
        cos = rope[..., 0]
        sin = rope[..., 1]
        dx1 = dy1 * cos + dy2 * sin
        dx2 = dy2 * cos - dy1 * sin

        dx = Tensor.stack([dx1, dx2], axis=-1)

        return dx.reshape(batch, seq_len, n_heads, head_dim)
