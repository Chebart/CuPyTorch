from __future__ import annotations
import contextlib
import operator

from .backend import AVAILABLE_BACKENDS

class Tensor:
    """
    Custom data class, supporting automatic backend selection (NumPy/CuPy) 
    and seamless CPU/GPU device tracking.
    """

    def __init__(self, data, dtype: str = "fp32", device: str = "cpu"):
        self._backend = Tensor.define_backend(device)
        self._dtype = Tensor.get_backend_dtype(self.backend, dtype)
        self._device = device

        if self.backend.__name__ == "cupy":
            with self.device_context():
                self._data = self.backend.asarray(data, dtype = self._dtype)
        elif self.backend.__name__ == "numpy":
            self._data = self.backend.array(data, dtype = self._dtype)

        # change dtype on string value
        self._dtype = dtype
        # set strides
        self._strides = self.data.strides

    # ---------------------
    # Verification methods
    # ---------------------

    def same_device(self, other: Tensor):
        return self.device == other.device

    @staticmethod
    def define_backend(device: str):
        if device.startswith("cuda"):
            if "cupy" not in AVAILABLE_BACKENDS:
                raise ValueError(f"CuPy is not installed. Cannot move tensor to {device}.")
            return AVAILABLE_BACKENDS["cupy"]
        
        return AVAILABLE_BACKENDS["numpy"]

    @staticmethod
    def get_backend_dtype(backend, dtype: str):
        dtype_map = {
            "fp16": backend.float16,
            "fp32": backend.float32,
            "fp64": backend.float64,
            "int16": backend.int16,
            "int32": backend.int32,
            "int64": backend.int64,
            "bool": backend.bool_,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: '{dtype}'")
        
        return dtype_map[dtype]
    
    # ---------------------
    # Properties
    # ---------------------

    @property
    def backend(self):
        return self._backend
    
    @property
    def data(self):
        return self._data

    @property
    def T(self):
        return Tensor(self.data.T, self.dtype, self.device)

    @property
    def device(self):
        return self._device
    
    @property
    def device_id(self):
        if self.device.startswith("cuda"): 
            return int(self.device.split(':')[-1])
        else:
            return None
    
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return len(self.data.shape)

    @property
    def strides(self):
        return self._strides

    @property
    def dtype(self):
        return self._dtype

    # ---------------------
    # Device transfer methods
    # ---------------------

    def to_device(self, device: str)-> Tensor:
        new_backend = self.define_backend(device)
        self._device = device

        if device.startswith("cuda"):
            if self.backend.__name__ == "numpy":
                self._backend = new_backend

            with self.backend.cuda.Device(self.device_id):
                self._data = self.backend.asarray(self.data)
        else:
            if self.backend.__name__ == "cupy":
                self._data = self.data.get()
                self._backend = new_backend

        return self

    # ---------------------
    # Context manager protocol
    # ---------------------

    @contextlib.contextmanager
    def device_context(self):
        if self.backend.__name__ == "cupy":
            with self.backend.cuda.Device(self.device_id):
                yield
        elif self.backend.__name__ == "numpy":
            yield

    # ---------------------
    # Binary operation methods
    # ---------------------

    def _binary_op(op_func, inplace: bool = False, matmul: bool = False):
        def method(self, other):
            if isinstance(other, Tensor):
                if not self.same_device(other):
                    raise ValueError(
                        f"Expected all tensors to be on the same device, "
                        f"but found {self.device} and {other.device}!"
                    )
                other = other.data

            with self.device_context():
                if matmul:
                    result = self.backend.matmul(self.data, other)
                else:
                    result = op_func(self.data, other)

            if inplace:
                self._data = result
                return self
            else:
                return Tensor(result, self.dtype, self.device)
        
        return method
    
    # Out-place ops
    __add__ = _binary_op(operator.add, inplace = False, matmul = False)
    __sub__ = _binary_op(operator.sub, inplace = False, matmul = False)
    __mul__ = _binary_op(operator.mul, inplace = False, matmul = False)
    __truediv__ = _binary_op(operator.truediv, inplace = False, matmul = False)
    __pow__ = _binary_op(operator.pow, inplace = False, matmul = False)
    __radd__ = __add__
    __rsub__ = _binary_op(lambda x, y: operator.sub(y, x), inplace = False, matmul = False)
    __rmul__ = __mul__
    __rtruediv__ = _binary_op(lambda x, y: operator.truediv(y, x), inplace = False, matmul = False)
    __rpow__ = _binary_op(lambda x, y: operator.pow(y, x), inplace = False, matmul = False)

    # In-place ops
    __iadd__ = _binary_op(operator.iadd, inplace = True, matmul = False)
    __isub__ = _binary_op(operator.isub, inplace = True, matmul = False)
    __imul__ = _binary_op(operator.imul, inplace = True, matmul = False)
    __itruediv__ = _binary_op(operator.itruediv, inplace = True, matmul = False)
    __ipow__ = _binary_op(operator.ipow, inplace = True, matmul = False)

    # Comparison ops
    __gt__ = _binary_op(lambda a, b: a > b, inplace = False, matmul = False)
    __lt__ = _binary_op(lambda a, b: a < b, inplace = False, matmul = False)
    __ge__ = _binary_op(lambda a, b: a >= b, inplace = False, matmul = False)
    __le__ = _binary_op(lambda a, b: a <= b, inplace = False, matmul = False)
    __eq__ = _binary_op(lambda a, b: a == b, inplace = False, matmul = False)
    __ne__ = _binary_op(lambda a, b: a != b, inplace = False, matmul = False)

    # Matrix multiplier ops
    __matmul__ = _binary_op(None, inplace = False, matmul = True)
    __rmatmul__ = _binary_op(None, inplace = False, matmul = True)

    # ---------------------
    # Unary operation methods
    # ---------------------

    def __neg__(self)-> Tensor:
        return Tensor(-self.data, self.dtype, self.device)

    def __abs__(self)-> Tensor:
        return Tensor(abs(self.data), self.dtype, self.device)

    # ---------------------
    # Elementwise methods
    # ---------------------

    def exp(self)-> Tensor:
        return Tensor(self.backend.exp(self.data), self.dtype, self.device)

    def log(self)-> Tensor:
        return Tensor(self.backend.log(self.data), self.dtype, self.device)

    def sign(self)-> Tensor:
        return Tensor(self.backend.sign(self.data), self.dtype, self.device)

    def sqrt(self):
        return Tensor(self.backend.sqrt(self.data), self.dtype, self.device)

    def tanh(self)-> Tensor:
        return Tensor(self.backend.tanh(self.data), self.dtype, self.device)

    def sin(self)-> Tensor:
        return Tensor(self.backend.sin(self.data), self.dtype, self.device)
    
    def cos(self)-> Tensor:
        return Tensor(self.backend.cos(self.data), self.dtype, self.device)

    # ---------------------
    # Views methods
    # ---------------------

    def __getitem__(self, index) -> Tensor:
        with self.device_context():
            # Handle multi-axis indexing
            if isinstance(index, tuple):
                processed_index = []
                for idx in index:
                    if isinstance(idx, Tensor):
                        idx_data = idx.data
                        if idx_data.dtype == self.backend.bool_:
                            processed_index.append(idx_data)
                        else:
                            processed_index.append(idx_data.astype(int))
                    else:
                        processed_index.append(idx)

                data_indexed = self.data[tuple(processed_index)]

            # Handle single Tensor index
            elif isinstance(index, Tensor):
                index_data = index.data
                if index_data.dtype == self.backend.bool_:
                    data_indexed = self.data[index_data]
                else:
                    data_indexed = self.data[index_data.astype(int)]

            # Handle normal Python indexing
            else:
                data_indexed = self.data[index]

            return Tensor(data_indexed, dtype=self.dtype, device=self.device)

    def __setitem__(self, index, value):
        with self.device_context():
            if isinstance(value, Tensor):
                value = value.data

            # Handle multi-axis indexing
            if isinstance(index, tuple):
                processed_index = []
                for idx in index:
                    if isinstance(idx, Tensor):
                        idx_data = idx.data
                        if idx_data.dtype == self.backend.bool_:
                            processed_index.append(idx_data)
                        else:
                            processed_index.append(idx_data.astype(int))
                    else:
                        processed_index.append(idx)

                self._data[tuple(processed_index)] = value

            # Handle single Tensor index
            elif isinstance(index, Tensor):
                idx_data = index.data
                if idx_data.dtype == self.backend.bool_:
                    self.data[idx_data] = value
                else:
                    self.data[idx_data.astype(int)] = value

            # Handle normal Python indexing
            else:
                self.data[index] = value

    def reshape(self, *shape)-> Tensor:
        return Tensor(self.data.reshape(*shape), self.dtype, self.device)

    def transpose(self, *axes)-> Tensor:
        return Tensor(self.data.transpose(*axes), self.dtype, self.device)

    def as_strided(self, shape, strides)-> Tensor:
        return Tensor(
            self.backend.lib.stride_tricks.as_strided(self.data, shape=shape, strides=strides), 
            self.dtype, 
            self.device
        )

    def put_along_axis(self, indices, values, axis):
        if isinstance(indices, Tensor):
            indices = indices.data.astype(int)
        if isinstance(values, Tensor):
            values = values.data

        self.backend.put_along_axis(self.data, indices, values, axis=axis)

    # ---------------------
    # Aggregation methods
    # ---------------------

    def max(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.max(axis=axis, keepdims=keepdims), self.dtype, self.device)

    def min(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.min(axis=axis, keepdims=keepdims), self.dtype, self.device)

    def sum(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.dtype, self.device)

    def mean(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.mean(axis=axis, keepdims=keepdims), self.dtype, self.device)

    def argmax(self, axis=None, keepdims=False)-> Tensor:
        return Tensor(self.data.argmax(axis=axis), self.dtype, self.device)

    # ---------------------
    # Utility methods
    # ---------------------

    def __len__(self):
        return self.shape[0]

    def fill(self, value: int):
        self.data.fill(value)
        return self

    def masked_fill(self, mask, value):
        if isinstance(mask, Tensor):
            mask = mask.data.astype(bool)
        out = self.backend.where(mask, value, self.data)
        return Tensor(out, dtype = self.dtype, device = self.device)
    
    def clone(self)-> Tensor:
        return Tensor(self.data.copy(), self.dtype, self.device)

    def clip(self, min_val: float, max_val: float):
        self.backend.clip(self.data, a_min = min_val, a_max = max_val, out = self.data)
        return self

    @staticmethod
    def pad(array, pad_width, mode = "constant", dtype = "fp16", device="cpu")-> Tensor:
        if isinstance(array, Tensor):
            array = array.data
        backend = Tensor.define_backend(device)
        return Tensor(backend.pad(array, pad_width, mode = mode), dtype, device)

    @staticmethod
    def concat(arr: list, axis: int, dtype = "fp16", device="cpu")-> Tensor:
        for i, element in enumerate(arr):
            if isinstance(element, Tensor):
                arr[i] = element.data

        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)
        return Tensor(backend.concatenate(arr, axis = axis, dtype = backend_dtype), dtype, device)
    
    @staticmethod
    def stack(arr: list, axis: int, dtype = "fp16", device="cpu")-> Tensor:
        for i, element in enumerate(arr):
            if isinstance(element, Tensor):
                arr[i] = element.data

        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)
        return Tensor(backend.stack(arr, axis = axis, dtype = backend_dtype), dtype, device)

    @staticmethod
    def zeros(shape, dtype = "fp16", device = "cpu")-> Tensor:
        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)
        return Tensor(backend.zeros(shape, dtype = backend_dtype), dtype, device)
    
    @staticmethod
    def ones(shape, dtype = "fp16", device = "cpu")-> Tensor:
        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)
        return Tensor(backend.ones(shape, dtype = backend_dtype), dtype, device)
    
    @staticmethod
    def eye(
        n: int, 
        m: int = None, 
        dtype = "fp16", 
        device: str = "cpu"
    )-> Tensor:
        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)
        return Tensor(backend.eye(n, m, dtype = backend_dtype), dtype, device)

    @staticmethod
    def where(
        condition, 
        x, 
        y,
        dtype = "fp16", 
        device: str = "cpu"        
    )-> Tensor:
        if isinstance(condition, Tensor):
            condition = condition.data
        if isinstance(x, Tensor):
            x = x.data
        if isinstance(y, Tensor):
            y = y.data

        backend = Tensor.define_backend(device)
        return Tensor(backend.where(condition, x, y), dtype, device)

    @staticmethod
    def maximum(
        x1, 
        x2,
        dtype = "fp16", 
        device: str = "cpu"          
    )-> Tensor:
        if isinstance(x1, Tensor):
            x1 = x1.data
        if isinstance(x2, Tensor):
            x2 = x2.data

        backend = Tensor.define_backend(device)
        return Tensor(backend.maximum(x1, x2), dtype, device)
    
    @staticmethod
    def minimum(
        x1, 
        x2,
        dtype = "fp16", 
        device: str = "cpu"          
    )-> Tensor:
        if isinstance(x1, Tensor):
            x1 = x1.data
        if isinstance(x2, Tensor):
            x2 = x2.data

        backend = Tensor.define_backend(device)
        return Tensor(backend.minimum(x1, x2), dtype, device)

    @staticmethod
    def random_uniform(
        low: float = 0.0, 
        high: float = 1.0, 
        size: tuple = (2, 2),
        dtype = "fp16",
        device: str = "cpu"
    )-> Tensor:
        backend = Tensor.define_backend(device)
        return Tensor(backend.random.uniform(low, high, size = size), dtype, device)

    @staticmethod
    def random_normal(
        mean: float = 0.0, 
        std: float = 1.0, 
        size: tuple = (2, 2), 
        dtype = "fp16", 
        device: str = "cpu"
    )-> Tensor:
        backend = Tensor.define_backend(device)
        return Tensor(backend.random.normal(mean, std, size = size), dtype, device)

    @staticmethod
    def rand(
        shape: tuple,
        dtype = "fp16", 
        device: str = "cpu"
    )-> Tensor:
        backend = Tensor.define_backend(device)
        return Tensor(backend.random.rand(*shape), dtype, device)

    @staticmethod
    def arange(
        start: float, 
        stop: float = None, 
        step: float = 1, 
        dtype: str = "fp32", 
        device: str = "cpu"
    ) -> Tensor:
        # set stop if needed
        if stop is None:
            stop = start
            start = 0

        backend = Tensor.define_backend(device)
        return Tensor(backend.arange(start, stop, step), dtype=dtype, device=device)

    @staticmethod
    def einsum(
        subscripts: str,
        *operands: Tensor,
        dtype: str = "fp32", 
        device: str = "cpu"
    ) -> Tensor:
        backend = Tensor.define_backend(device)
        arrays = [op.data for op in operands]
        return Tensor(backend.einsum(subscripts, *arrays), dtype, device)

    # ---------------------
    # Data type conversion methods
    # ---------------------

    def to_numpy(self):
        if self.backend.__name__ == "cupy":
            return self.data.get()
        elif self.backend.__name__ == "numpy":
            return self.data

    def astype(self, dtype = "fp16"):
        backend_dtype = Tensor.get_backend_dtype(self.backend, dtype)
        self._dtype = dtype
        if self.backend.__name__ == "cupy":
            with self.device_context():
                self._data = self.backend.asarray(self.data, dtype = backend_dtype)
        elif self.backend.__name__ == "numpy":
            self._data = self.backend.array(self.data, dtype = backend_dtype)

        return self

    # ---------------------
    # Representation methods
    # ---------------------

    def __repr__(self):
        return f"Tensor({self.data}, device={self.device})"