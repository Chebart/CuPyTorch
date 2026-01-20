from .abstract_block import AbstractBlock
from .init_weights import xavier 
from core.data import Tensor

class GCNConv(AbstractBlock):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        dtype: str = "fp32",
        bias: bool = True
    ):
        # Set bias flag
        self._bias = bias
        # Init trainable params and small increments
        self._w = xavier((out_features, in_features), dtype = dtype, uniform = True)
        self._b = Tensor.zeros((out_features), dtype = dtype)
        self._dw = Tensor.zeros(self._w.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)

    def normalize_adjacency(self, adj):
        # Add self-loops: A_hat = A + I
        N = adj.shape[0]
        I = Tensor.eye(N, dtype=adj.dtype, device=adj.device)
        adj_hat = adj + I

        # Calculate degree matrix = D_hat**(-0.5)
        degree = adj_hat.sum(axis=1)
        deg_inv_sqrt = degree ** -0.5
        deg_inv_sqrt = deg_inv_sqrt.masked_fill(deg_inv_sqrt.isinf(), 0.0)
        D_inv_sqrt = Tensor.diag(deg_inv_sqrt, dtype=adj.dtype, device=adj.device)

        return D_inv_sqrt @ adj_hat @ D_inv_sqrt

    def forward(self, x, adj):
        # Cache input
        self.x = x

        # Normalize adjency
        self.adj_norm = self.normalize_adjacency(adj)
        # Do linear transformation
        self.support = x @ self.W
        # Graph propagation
        out = self.adj_norm @ self.support + self._b

        return out

    def parameters(self):
        if self._bias:
            return [('w', self._w, self._dw), ('b', self._b, self._db)]
        else:
            return [('w', self._w, self._dw)]

    def backward(self, dLdy):
        self._db += dLdy.sum(axis = 0)
        self._dw += self.adj_norm @ self.x @ dLdy
        return self.adj_norm @ dLdy @ self._w.T