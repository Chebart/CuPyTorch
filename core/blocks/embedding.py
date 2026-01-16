from .abstract_block import AbstractBlock
from ..data import Tensor

class Embedding(AbstractBlock):
    """Embedding(x) = E[x]"""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        weights: Tensor | None = None,
        dtype: str = "fp32"
    ):
        # Create embeddings and small increments
        if weights is not None:
            assert weights.shape == (vocab_size, embedding_dim), \
                f"weights must have shape ({vocab_size}, {embedding_dim})"
            self._embeddings = weights
        else:
            self._embeddings = Tensor.rand((vocab_size, embedding_dim), dtype=dtype)

        self._dembeddings = Tensor.zeros(self._embeddings.shape, dtype=dtype)

    def forward(self, x):
        self.x = x
        out = Tensor.zeros(
            (len(x), self._embeddings.shape[1]),
            dtype = x.dtype,
            device = x.device
        )
        for i, token_idx in enumerate(x):
            out[i] = self._embeddings[token_idx]

        return out

    def parameters(self):
        return [("emb", self._embeddings, self._dembeddings)]

    def backward(self, dLdy):
        for i, token_idx in enumerate(self.x):
            self._dembeddings[token_idx] += dLdy[i]