from .abstract_block import AbstractBlock
from core.data import Tensor

class BatchNorm2d(AbstractBlock):
    """
    Works per-channel using batch statistics
    BatchNorm2d(x) = (x - mean_c) / sqrt(var_c + eps) * gamma_c + beta_c
    """

    def __init__(
        self, 
        num_features, 
        eps: float = 1e-5,
        dtype: str = "fp32"
    ):
        self.num_features = num_features
        self.eps = eps
        # Init trainable params and small increments
        self._g = Tensor.ones((num_features), dtype = dtype)
        self._b = Tensor.zeros((num_features), dtype = dtype)
        self._dg = Tensor.zeros(self._g.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)
    
    def forward(self, x):
        self.x = x
        self.m = x.mean(axis=(0, 2, 3), keepdims=True)
        self.v = ((x - self.m)**2).mean(axis=(0, 2, 3), keepdims=True)
        self.inv_std = 1.0 / (self.v + self.eps).sqrt()
        self.x_scaled = (x - self.m) * self.inv_std

        g = self._g.reshape(1, -1, 1, 1)
        b = self._b.reshape(1, -1, 1, 1)

        return self.x_scaled * g + b
        
    def parameters(self):
        return [('b', self._g, self._dg), ('b', self._b, self._db)]

    def backward(self, dLdy):
        N, C, H, W = dLdy.shape
        M = N * H * W

        self._db += dLdy.sum(axis=(0, 2, 3))
        self._dg += (dLdy * self.x_scaled).sum(axis=(0, 2, 3))

        g = self._g.reshape(1, C, 1, 1)
        dLdx_scaled = dLdy * g
        dLdv = (dLdx_scaled * (self.x - self.m) * -0.5 * self.inv_std**3).sum(axis=(0, 2, 3), keepdims = True)
        dLdm_part1 = (dLdx_scaled * -self.inv_std).sum(axis=(0, 2, 3), keepdims = True)
        dLdm_part2 = dLdv * (-(2 / M) * (self.x - self.m).sum(axis=(0, 2, 3), keepdims = True))
        dLdm = dLdm_part1 + dLdm_part2
        dLdx = dLdx_scaled * self.inv_std + dLdv * 2 * (self.x - self.m) / M + dLdm / M

        return dLdx