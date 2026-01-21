from .abstract_block import AbstractBlock
from .sigmoid import Sigmoid
from .tanh import Tanh
from .init_weights import xavier 
from core.data import Tensor

class LSTMBlock(AbstractBlock):
    """
    i_t = σ(x_t * W_ii.T + b_ii + h_t-1 * W_hi.T + b_hi)        
    f_t = σ(x_t * W_if.T + b_if + h_t-1 * W_hf.T + b_hf)        
    g_t = tanh(x_t * W_ig.T + b_ig + h_t-1 * W_hg.T + b_hg)     
    o_t = σ(x_t * W_io.T + b_io + h_t-1 * W_ho.T + b_ho)      
    c_t = f_t * c_t-1 + i_t * g_t                               
    h_t = o_t * tanh(c_t)                                   
    """

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
        self._w = xavier((4 * hidden_size, input_size + hidden_size), dtype = dtype, uniform = uniform_init)
        self._b = Tensor.zeros((4 * hidden_size), dtype = dtype)
        self._dw = Tensor.zeros(self._w.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)
        # Init needed activation functions
        self._i_by_t = []
        self._f_by_t = []
        self._g_by_t = []
        self._o_by_t = []
        self._c_by_t = []
        self._c_prev_by_t = []
        self._combined_by_t = []
        
    def forward(self, inp, h_prev, c_prev):
        # create activations for timestamp
        i_sigmoid = Sigmoid()
        f_sigmoid = Sigmoid()
        g_tanh = Tanh()
        o_sigmoid = Sigmoid()
        c_tanh = Tanh()

        # calcute hidden states
        combined = Tensor.concat([inp, h_prev], axis = 1, dtype = inp.dtype, device = inp.device)
        gates = combined @ self._w.T + self._b
        
        i = i_sigmoid(gates[:, :1*self.hidden_size])
        f = f_sigmoid(gates[:, 1*self.hidden_size:2*self.hidden_size])
        o = o_sigmoid(gates[:, 2*self.hidden_size:3*self.hidden_size])
        g = g_tanh(gates[:, 3*self.hidden_size:])

        # update long and short memory
        c_next = f * c_prev + i * g
        c = c_tanh(c_next)
        h_next = o * c

        # save states
        self._i_by_t.append(i_sigmoid)
        self._f_by_t.append(f_sigmoid)
        self._g_by_t.append(g_tanh)
        self._o_by_t.append(o_sigmoid)
        self._c_by_t.append(c_tanh)
        self._c_prev_by_t.append(c_prev)
        self._combined_by_t.append(combined)

        return h_next, c_next
   
    def parameters(self):
        if self._bias:
            return [('w', self._w, self._dw), ('b', self._b, self._db)]
        else:
            return [('w', self._w, self._dw)]
 
    def reset_cache(self):
        self._i_by_t = []
        self._f_by_t = []
        self._g_by_t = []
        self._o_by_t = []
        self._c_by_t = []
        self._c_prev_by_t = []
        self._combined_by_t = []

    def backward(self, dh_next, dLdc_next, t):
        dLdh_next = self._o_by_t[t].y * self._c_by_t[t].backward(dh_next)
        dLdc = dLdh_next + dLdc_next

        dLdi = self._i_by_t[t].backward(dLdc * self._g_by_t[t].y)
        dLdf = self._f_by_t[t].backward(dLdc * self._c_prev_by_t[t])
        dLdo = self._o_by_t[t].backward(dh_next * self._c_by_t[t].y)
        dLdg = self._g_by_t[t].backward(dLdc * self._i_by_t[t].y)

        dGates = Tensor.concat([dLdi, dLdf, dLdo, dLdg], axis = 1, dtype = dLdi.dtype, device = dLdi.device)
        self._dw += dGates.T @ self._combined_by_t[t]
        self._db += dGates.sum(axis=0)
        
        self._dcombined = dGates @ self._w
        dLdx = self._dcombined[:, :self.input_size]
        dLdh_prev = self._dcombined[:, self.input_size:]
        dLdc_prev = dLdc * self._f_by_t[t].y

        return dLdx, dLdh_prev, dLdc_prev