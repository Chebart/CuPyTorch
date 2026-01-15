from .abstract_block import AbstractBlock
from core.data import Tensor
from .tanh import Tanh

class GELU(AbstractBlock):
    """GELU(x)= 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3 )))"""
    
    def forward(self, x):
        self.tanh = Tanh()
        self.x = x
        self.scale = Tensor((2 / 3.14) ** 0.5, dtype=x.dtype, device=x.device)
        self.tanh_res = self.tanh(self.scale * (x + 0.044715 * x**3))
        self.y = 0.5 * x * (1 + self.tanh_res)
        return self.y

    def parameters(self):
        return []

    def backward(self, dLdy):
        dLdt = dLdy * 0.5 * self.x
        dLdu = self.tanh.backward(dLdt)
        du_dx = self.scale * (1 + 3 * 0.044715 * self.x**2)
        return dLdy * 0.5 * (1 + self.tanh_res) + dLdu * du_dx