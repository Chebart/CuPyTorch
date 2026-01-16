from .abstract_block import AbstractBlock
from ..data import Tensor

class PositionalEncoding(AbstractBlock):
    """PositionalEncoding(x) = x + PE(pos, d_model)"""

    def __init__(
        self, 
        d_model: int, 
        max_seq_length: int,
        dtype: str = "fp32"
    ):
        self.max_seq_len = max_seq_length
        # Create positional encodings
        pe = Tensor.zeros((max_seq_length, d_model), dtype = dtype)
        position = Tensor.arange(max_seq_length, dtype = dtype)[:, None]
        div_term = (Tensor.arange(0, d_model, 2, dtype = dtype) * \
                    -(Tensor(10000.0, dtype = dtype).log() / d_model)).exp()

        value = position * div_term
        pe[:, 0::2] = value.sin()
        pe[:, 1::2] = value.cos()

        # Add batch dimension
        self.pe = pe[None, :, :]
        
    def forward(self, x):
        assert x.shape[1] <= self.max_seq_len, (
            f"PE seq_len mismatch: seq_len={x.shape[1]}, max_seq_len={self.max_seq_len}"
        )
        return x + self.pe[:, :x.shape[1], :].to_device(x.device)

    def parameters(self):
        return []

    def backward(self, dLdy):
        return dLdy