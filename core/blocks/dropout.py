from .abstract_block import AbstractBlock
from core.data import Tensor

class Dropout(AbstractBlock):
    """Dropout(x, p) = x * mask / (1 - p)"""
    
    def __init__(
        self,
        p: float
    ):
        self.p = p
        self.is_train = True

    def forward(self, x):
        if self.is_train:
            self.mask = (Tensor.rand(x.shape, dtype = x.dtype, device = x.device) >= self.p).astype(x.dtype)
            self.mask /= (1.0 - self.p)
            return x * self.mask
        else:
            return x
        
    def parameters(self):
        return []

    def backward(self, dLdy):
        if self.is_train:
            return dLdy * self.mask
        else:
            return dLdy