from .abstract_block import AbstractBlock
from .sigmoid import Sigmoid
from .tanh import Tanh
from .init_weights import xavier 
from core.data import Tensor

class GRUBlock(AbstractBlock):
    """
    r_t = σ(x_t * W_ir.T + b_ir + h_t-1 * W_hr.T + b_hr)
    z_t = σ(x_t * W_iz.T + b_iz + h_t-1 * W_hz.T + b_hz)
    n_t = tanh(x_t * W_in.T + b_in + (r_t * (h_t-1 * W_hn.T + b_hn)))
    h_t = (1 − z_t) * n_t + z_t * h_t-1                            
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
        self._w = xavier((3 * hidden_size, input_size + hidden_size), dtype = dtype, uniform = uniform_init)
        self._b = Tensor.zeros((3 * hidden_size), dtype = dtype)
        self._dw = Tensor.zeros(self._w.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)
        # Init needed activation functions
        self._r_by_t = []
        self._z_by_t = []
        self._h_tilda_by_t = []
        self._combined_by_t = []
        
    def forward(self, inp, h_prev):
        # create activations for timestamp
        r_sigmoid = Sigmoid()
        z_sigmoid = Sigmoid()
        h_tilda_tanh = Tanh()

        # calcute hidden states
        combined = Tensor.concat([inp, h_prev], axis = 1, dtype = inp.dtype, device = inp.device)
        gates = combined @ self._w.T + self._b
        
        z = z_sigmoid(gates[:, :1*self.hidden_size])
        r = r_sigmoid(gates[:, 1*self.hidden_size:2*self.hidden_size])
        h_tilda = gates[:, 2*self.hidden_size:] + r * combined[:, self.input_size:]
        h_tilda = h_tilda_tanh(h_tilda)

        # update hidden state
        h_next = (1 - z) * h_tilda + z * h_prev

        # save states
        self._r_by_t.append(r_sigmoid)
        self._z_by_t.append(z_sigmoid)
        self._h_tilda_by_t.append(h_tilda_tanh)
        self._combined_by_t.append(combined)

        return h_next
   
    def parameters(self):
        if self._bias:
            return [('w', self._w, self._dw), ('b', self._b, self._db)]
        else:
            return [('w', self._w, self._dw)]
 
    def reset_cache(self):
        self._r_by_t = []
        self._z_by_t = []
        self._h_tilda_by_t = []
        self._combined_by_t = []

    def backward(self, dh_next, t):
        r = self._r_by_t[t].y
        z = self._z_by_t[t].y
        dLdh_tilda = self._h_tilda_by_t[t].backward(dh_next * (1 - z))

        h_prev = self._combined_by_t[t][:, self.input_size:]
        dLdr = dLdh_tilda * h_prev * r * (1 - r)
        dLdz = dh_next * (h_prev - self._h_tilda_by_t[t].y) * z * (1 - z)

        dGates = Tensor.concat([dLdz, dLdr, dLdh_tilda], axis = 1, dtype = dLdr.dtype, device = dLdr.device)
        self._dw += dGates.T @ self._combined_by_t[t]
        self._db += dGates.sum(axis=0)

        self._dcombined = dGates @ self._w
        dLdx = self._dcombined[:, :self.input_size]
        dLdh_prev_part = dh_next * z
        dLdh_prev = self._dcombined[:, self.input_size:] + dLdh_prev_part

        return dLdx, dLdh_prev