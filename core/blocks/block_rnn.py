from .abstract_block import AbstractBlock
from .tanh import Tanh
from .init_weights import xavier 
from core.data import Tensor

class RNNBlock(AbstractBlock):
    """h_t = tanh(x_t · W_ih.T + b_ih + h_t-1 · W_hh.T + b_hh)"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dtype: str = "fp32",
        bias: bool = True,
        uniform_init: bool = False
    ):
        # Remember dims
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Set bias flag
        self._bias = bias
        # Init trainable params and small increments
        self._w = xavier((hidden_size, input_size + hidden_size), dtype = dtype, uniform = uniform_init)
        self._b = Tensor.zeros((hidden_size), dtype = dtype)
        self._dw = Tensor.zeros(self._w.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)
        # Init needed activation functions
        self._tanh_by_t = []
        self._combined_by_t = []
        
    def forward(self, inp, h_prev):
        # create tanh for timestamp
        tanh = Tanh()
        # calcute hidden state
        combined = Tensor.concat([inp, h_prev], axis = 1, dtype = inp.dtype, device = inp.device)
        y = tanh(combined @ self._w.T + self._b)
        # save states
        self._tanh_by_t.append(tanh)
        self._combined_by_t.append(combined)
        return y
    
    def parameters(self):
        if self._bias:
            return [('w', self._w, self._dw), ('b', self._b, self._db)]
        else:
            return [('w', self._w, self._dw)]
 
    def reset_cache(self):
        self._tanh_by_t = []
        self._combined_by_t = []     

    def backward(self, dLdy, t):
        self._dtanh = self._tanh_by_t[t].backward(dLdy)

        self._dw += (self._dtanh.T @ self._combined_by_t[t]).clip(-1, 1)
        self._db += (self._dtanh.sum(axis=0)).clip(-1, 1)
        self._dcombined = self._dtanh @ self._w

        dLdx = self._dcombined[:, :self.input_size]
        dLdh_prev = self._dcombined[:, self.input_size:]

        return dLdx, dLdh_prev