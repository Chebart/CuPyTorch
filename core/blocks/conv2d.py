from ..utils import split_2d_data_on_windows, get_2d_data_from_windows
from .abstract_block import AbstractBlock
from .init_weights import xavier 
from core.data import Tensor

class Conv2D(AbstractBlock):
    """out[N_i, C_out] = bias[C_out] + sum_{k=0}^{C_in-1} weight[C_out, k] * input[N_i, k]"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        uniform_init: bool = False,
        dtype: str = "fp32"
    ):
        # Init hyperparams
        self._in = in_channels
        self._out = out_channels
        self._k = kernel_size
        self._s = stride
        self._p = padding
        self._bias = bias

        # Init trainable params and small increments
        self._w = xavier((self._out, self._in, kernel_size, kernel_size), dtype = dtype, uniform = uniform_init)
        self._b = Tensor.zeros((self._out), dtype = dtype)
        self._dw = Tensor.zeros(self._w.shape, dtype = dtype)
        self._db = Tensor.zeros(self._b.shape, dtype = dtype)

    def forward(self, x):
        # Check input dims
        if x.ndim != 4:
            raise RuntimeError("x must be 4d array with shape (N, B, H, W)")

        self.x = x
        # Get windows
        windows, _ = split_2d_data_on_windows(x, k_size = self._k, stride = self._s, pad = self._p)
        self.windows = windows

        # Reshape windows
        N, C, H_out, W_out, _, _ = windows.shape
        self.x_col = windows.reshape(N, C, H_out, W_out, -1)
        self.x_col = self.x_col.transpose(0, 2, 3, 1, 4)
        self.x_col = self.x_col.reshape(N*H_out*W_out, -1)  
        # Reshape weigths
        self.w_col = self._w.reshape(self._out, -1)
        # Get result
        out = (self.x_col @ self.w_col.T + self._b).reshape(N, H_out, W_out, self._out).transpose(0, 3, 1, 2)
        return out

    def parameters(self):
        if self._bias:
            return [('w', self._w, self._dw), ('b', self._b, self._db)]
        else:
            return [('w', self._w, self._dw)]

    def backward(self, dLdy):
        # dLdy shape is (N, out_channels, H_out, W_out)
        self._db += dLdy.sum(axis=(0, 2, 3))

        dLdy = dLdy.transpose(0, 2, 3, 1).reshape(-1, self._out)
        self._dw += (dLdy.T @ self.x_col).reshape(self._w.shape)

        N, C, H_out, W_out, k_size, _ = self.windows.shape
        dLdx_col = (dLdy @ self.w_col).reshape(N, H_out, W_out, C, k_size, k_size)
        dLdx_col = dLdx_col.transpose(0, 3, 1, 2, 4, 5)
        dLdx = get_2d_data_from_windows(dLdx_col, stride = self._s, pad = self._p)

        return dLdx