from math import prod

from .abstract_block import AbstractBlock
from core.data import Tensor

class LayerNorm(AbstractBlock):
    """
    Works per-sample over the last D dimensions
    LayerNorm(x) = (x - mean_c) / sqrt(var_c + eps) * gamma_c + beta_c
    """

    def __init__(
        self, 
        normalized_shape, 
        eps: float = 1e-5,
        dtype: str = "fp32"
    ):
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.axes = tuple(range(-len(self.normalized_shape), 0))
        # Init trainable params and small increments
        self._g = Tensor.ones(normalized_shape, dtype = dtype)
        self._b = Tensor.zeros(normalized_shape, dtype = dtype)
        self._dg = Tensor.zeros(self._g.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)
    
    def forward(self, x):
        self.x = x
        self.m = x.mean(axis=self.axes, keepdims=True)
        self.v = ((x - self.m)**2).mean(axis=self.axes, keepdims=True)
        self.inv_std = 1.0 / (self.v + self.eps).sqrt()
        self.x_scaled = (x - self.m) * self.inv_std

        param_shape = (1,) * (x.ndim - len(self.normalized_shape)) + self.normalized_shape
        g = self._g.reshape(param_shape)
        b = self._b.reshape(param_shape)

        return self.x_scaled * g + b
        
    def parameters(self):
        return [('norm', self._g, self._dg), ('norm', self._b, self._db)]

    def backward(self, dLdy):
        param_axes = (0,) + self.axes
        M = prod(self.normalized_shape)

        self._db += dLdy.sum(axis=param_axes)
        self._dg += (dLdy * self.x_scaled).sum(axis=param_axes)

        g_shape = (1,) * (dLdy.ndim - len(self.normalized_shape)) + self.normalized_shape
        g = self._g.reshape(g_shape)

        dLdx_scaled = dLdy * g
        dLdv = (dLdx_scaled * (self.x - self.m) * -0.5 * self.inv_std**3).sum(axis=self.axes, keepdims=True)
        dLdm_part1 = (dLdx_scaled * -self.inv_std).sum(axis=self.axes, keepdims = True)
        dLdm_part2 = dLdv * (-(2 / M) * (self.x - self.m).sum(axis=self.axes, keepdims = True))
        dLdm = dLdm_part1 + dLdm_part2
        dLdx = dLdx_scaled * self.inv_std + dLdv * 2 * (self.x - self.m) / M + dLdm / M

        return dLdx