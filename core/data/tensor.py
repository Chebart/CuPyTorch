from __future__ import annotations
import contextlib
import operator

from .backend import AVAILABLE_BACKENDS

class Tensor:
    """
    Custom data class, supporting automatic backend selection (NumPy/CuPy)
    and seamless CPU/GPU device tracking.
    """
    _initialized_cuda_devices = set()

    def __init__(self, data, dtype: str = "fp32", device: str = "cpu"):
        self._backend = Tensor.define_backend(device)
        self._dtype = dtype
        self._device = device

        # Init device if needed
        Tensor.init_device(device)

        # create array on the correct device
        backend_dtype = Tensor.get_backend_dtype(self.backend, dtype)
        if self.backend.__name__ == "cupy":
            self._data = self._run_on_device(lambda: self.backend.asarray(data, dtype=backend_dtype))
        else:
            self._data = self._run_on_device(lambda: self.backend.array(data, dtype=backend_dtype))

        # store strides lazily from data
        self._strides = self.data.strides

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
        out = self._run_on_device(lambda: self.data.T)
        return Tensor(out, self.dtype, self.device)

    @property
    def device(self):
        return self._device

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
    # Backend methods
    # ---------------------

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
    # Dtype methods
    # ---------------------

    @staticmethod
    def _dtype_rank(dtype: str) -> int:
        order = {
            "bool": 0,
            "int16": 1,
            "int32": 2,
            "int64": 3,
            "fp16": 4,
            "fp32": 5,
            "fp64": 6,
        }
        return order.get(dtype, 5)

    @staticmethod
    def _get_common_dtype(dtype1: str, dtype2: str) -> str:
        if dtype1 == dtype2:
            return dtype1
        
        r1 = Tensor._dtype_rank(dtype1)
        r2 = Tensor._dtype_rank(dtype2)
        return dtype1 if r1 >= r2 else dtype2

    # ---------------------
    # Device methods
    # ---------------------

    @staticmethod
    def init_device(device: str):
        # Check backend
        backend = Tensor.define_backend(device)
        if backend.__name__ != "cupy":
            return

        # Check device was already initialized
        dev_id = Tensor.device_id(device)
        if dev_id in Tensor._initialized_cuda_devices:
            return

        # Init device
        rt = backend.cuda.runtime
        rt.setDevice(dev_id)
        rt.free(0)
        Tensor._initialized_cuda_devices.add(dev_id)

    @staticmethod
    def device_id(device: str) -> int | None:
        if device.startswith("cuda"):
            return int(device.split(":")[-1])
        return None

    def same_device(self, other: Tensor):
        return self.device == other.device

    def to_device(self, device: str) -> Tensor:
        # If already on the correct device, do nothing
        if self.device == device:
            return self

        # Init device if needed
        Tensor.init_device(device)

        self._device = device
        new_backend = self.define_backend(device)
        old_backend = self.backend

        if device.startswith("cuda"):
            # moving to GPU
            self._backend = new_backend
            self._data = self._run_on_device(lambda: self.backend.asarray(self.data))
        else:
            # moving to CPU
            if old_backend.__name__ == "cupy":
                self._data = self.data.get()
                self._backend = new_backend
            else:
                self._backend = new_backend
                self._data = self.data

        return self

    # ---------------------
    # Context managers
    # ---------------------

    @staticmethod
    @contextlib.contextmanager
    def device_context(device: str):
        backend = Tensor.define_backend(device)
        if backend.__name__ == "cupy":
            dev_id = Tensor.device_id(device) or 0
            with backend.cuda.Device(dev_id):
                yield
        else:
            yield

    # ---------------------
    # Internal executors
    # ---------------------

    @staticmethod
    def _run_static_on_device(device: str, fn):
        with Tensor.device_context(device):
            return fn()

    def _run_on_device(self, fn):
        with Tensor.device_context(self.device):
            return fn()

    def _resolve_binary_operands(self, other):
        if isinstance(other, Tensor):
            # Tensor * Tensor
            if not self.same_device(other):
                raise ValueError(
                    f"Expected all tensors to be on the same device, "
                    f"but found {self.device} and {other.device}!"
                )
            dtype = Tensor._get_common_dtype(self.dtype, other.dtype)
            return other.data, dtype
        else:
            # Tensor * scalar/array
            result_dtype = self.dtype
            rhs = other

        return rhs, result_dtype

    def _as_backend_array(self, x, backend_dtype):
        # Unwrap Tensor
        if isinstance(x, Tensor):
            x = x.data

        # If x is scalar
        if isinstance(x, (int, float, bool)):
            return x

        # Must already be a backend array
        if not isinstance(x, self.backend.ndarray):
            raise TypeError(
                f"Expected {self.backend.__name__}.ndarray, got {type(x).__name__}"
            )

        def run():
            if self.backend.__name__ == "cupy":
                # if device mismatch, reallocate array
                if x.device.id != self.device_id:
                    return self.backend.asarray(x, dtype=backend_dtype)

            # fix dtype
            if x.dtype != backend_dtype:
                return x.astype(backend_dtype, copy=False)

            return x

        return self._run_on_device(run)

    def _apply_binary_op(self, other, op_func, *, inplace=False):
        # Resolve operands and dtype
        rhs_data, result_dtype = self._resolve_binary_operands(other)
        backend_dtype = Tensor.get_backend_dtype(self.backend, result_dtype)

        # Define kernel
        def run():
            a = self._as_backend_array(self.data, backend_dtype)
            b = self._as_backend_array(rhs_data, backend_dtype)
            return op_func(a, b)

        # Execute operation
        result = self._run_on_device(run)

        # Inplace data if needed
        if inplace:
            self._data = result
            self._dtype = result_dtype
            return self

        return Tensor(result, result_dtype, self.device)

    @staticmethod
    def _static_op(
        fn,
        *args,
        dtype="fp16",
        device="cpu",
    ) -> Tensor:
        
        # Init device if needed
        Tensor.init_device(device)

        backend = Tensor.define_backend(device)
        backend_dtype = Tensor.get_backend_dtype(backend, dtype)

        # unwrap Tensors
        arrays = [
            a.data if isinstance(a, Tensor) else a
            for a in args
        ]

        def run():
            out = fn(backend, backend_dtype, *arrays)
            if hasattr(out, "dtype") and out.dtype != backend_dtype:
                out = out.astype(backend_dtype, copy=False)

            return out

        out = Tensor._run_static_on_device(device, run)
        return Tensor(out, dtype=dtype, device=device)

    # ---------------------
    # Binary operation methods
    # ---------------------

    def _binary_op(op_func, inplace: bool = False):
        def method(self, other):
            return self._apply_binary_op(other, op_func, inplace=inplace)
        return method

    # Out-of-place arithmetic
    __add__ = _binary_op(operator.add)
    __sub__ = _binary_op(operator.sub)
    __mul__ = _binary_op(operator.mul)
    __truediv__ = _binary_op(operator.truediv)
    __pow__ = _binary_op(operator.pow)

    # Reversed arithmetic
    __radd__ = _binary_op(lambda a, b: b + a)
    __rsub__ = _binary_op(lambda a, b: b - a)
    __rmul__ = _binary_op(lambda a, b: b * a)
    __rtruediv__ = _binary_op(lambda a, b: b / a)
    __rpow__ = _binary_op(lambda a, b: b ** a)

    # In-place arithmetic
    __iadd__ = _binary_op(operator.iadd, inplace=True)
    __isub__ = _binary_op(operator.isub, inplace=True)
    __imul__ = _binary_op(operator.imul, inplace=True)
    __itruediv__ = _binary_op(operator.itruediv, inplace=True)
    __ipow__ = _binary_op(operator.ipow, inplace=True)

    # Comparisons
    __gt__ = _binary_op(operator.gt)
    __lt__ = _binary_op(operator.lt)
    __ge__ = _binary_op(operator.ge)
    __le__ = _binary_op(operator.le)
    __eq__ = _binary_op(operator.eq)
    __ne__ = _binary_op(operator.ne)

    # Matrix multiplication (cleanest solution)
    __matmul__ = _binary_op(lambda a, b: a @ b)
    __rmatmul__ = _binary_op(lambda a, b: b @ a)

    # ---------------------
    # Unary operation methods
    # ---------------------

    def __neg__(self) -> Tensor:
        out = self._run_on_device(lambda: -self.data)
        return Tensor(out, self.dtype, self.device)

    def __abs__(self) -> Tensor:
        out = self._run_on_device(lambda: abs(self.data))
        return Tensor(out, self.dtype, self.device)

    # ---------------------
    # Elementwise methods
    # ---------------------

    def exp(self) -> Tensor:
        out = self._run_on_device(lambda: self.backend.exp(self.data))
        return Tensor(out, self.dtype, self.device)

    def log(self) -> Tensor:
        out = self._run_on_device(lambda: self.backend.log(self.data))
        return Tensor(out, self.dtype, self.device)

    def sign(self) -> Tensor:
        out = self._run_on_device(lambda: self.backend.sign(self.data))
        return Tensor(out, self.dtype, self.device)

    def sqrt(self):
        out = self._run_on_device(lambda: self.backend.sqrt(self.data))
        return Tensor(out, self.dtype, self.device)

    def tanh(self) -> Tensor:
        out = self._run_on_device(lambda: self.backend.tanh(self.data))
        return Tensor(out, self.dtype, self.device)

    def sin(self) -> Tensor:
        out = self._run_on_device(lambda: self.backend.sin(self.data))
        return Tensor(out, self.dtype, self.device)

    def cos(self) -> Tensor:
        out = self._run_on_device(lambda: self.backend.cos(self.data))
        return Tensor(out, self.dtype, self.device)

    # ---------------------
    # Views methods
    # ---------------------

    def _normalize_index(self, index):
        if isinstance(index, Tensor):
            idx = index.data
            if idx.dtype == self.backend.bool_:
                return idx
            return idx.astype(int)

        if isinstance(index, tuple):
            return tuple(self._normalize_index(i) for i in index)

        return index

    def __getitem__(self, index) -> Tensor:
        def run():
            idx = self._normalize_index(index)
            return self.data[idx]

        out = self._run_on_device(run)
        return Tensor(out, dtype=self.dtype, device=self.device)

    def __setitem__(self, index, value):
        def run():
            idx = self._normalize_index(index)
            val = value.data if isinstance(value, Tensor) else value
            self._data[idx] = val

        self._run_on_device(run)

    def item(self):
        return self.data.item()

    def reshape(self, *shape) -> Tensor:
        out = self._run_on_device(lambda: self.data.reshape(*shape))
        return Tensor(out, self.dtype, self.device)

    def transpose(self, *axes) -> Tensor:
        out = self._run_on_device(lambda: self.data.transpose(*axes))
        return Tensor(out, self.dtype, self.device)

    def as_strided(self, shape, strides) -> Tensor:
        out = self._run_on_device(
            lambda: self.backend.lib.stride_tricks.as_strided(
                self.data, 
                shape=shape, 
                strides=strides
            )
        )
        return Tensor(out, self.dtype, self.device)

    def put_along_axis(self, indices, values, axis):
        indices = self._as_backend_array(indices, self.dtype).astype(int)
        values = self._as_backend_array(values, self.dtype)
        self._run_on_device(lambda: self.backend.put_along_axis(self.data, indices, values, axis=axis))

    # ---------------------
    # Aggregation methods
    # ---------------------

    def max(self, axis=None, keepdims=False) -> Tensor:
        out = self._run_on_device(lambda: self.data.max(axis=axis, keepdims=keepdims))
        return Tensor(out, self.dtype, self.device)

    def min(self, axis=None, keepdims=False) -> Tensor:
        out = self._run_on_device(lambda: self.data.min(axis=axis, keepdims=keepdims))
        return Tensor(out, self.dtype, self.device)

    def sum(self, axis=None, keepdims=False) -> Tensor:
        out = self._run_on_device(lambda: self.data.sum(axis=axis, keepdims=keepdims))
        return Tensor(out, self.dtype, self.device)

    def mean(self, axis=None, keepdims=False) -> Tensor:
        out = self._run_on_device(lambda: self.data.mean(axis=axis, keepdims=keepdims))
        return Tensor(out, self.dtype, self.device)

    def argmax(self, axis=None, keepdims=False) -> Tensor:
        out = self._run_on_device(lambda: self.data.argmax(axis=axis, keepdims=keepdims))
        return Tensor(out, self.dtype, self.device)

    def any(self, axis=None, keepdims=False) -> Tensor:
        out = self._run_on_device(lambda: self.data.any(axis=axis, keepdims=keepdims))
        return Tensor(out, self.dtype, self.device)

    # ---------------------
    # Utility methods
    # ---------------------

    def __len__(self):
        return self.shape[0]

    def fill(self, value: int):
        self._run_on_device(lambda: self.data.fill(value))
        return self

    def masked_fill(self, mask, value):
        mask = self._as_backend_array(mask, self.dtype).astype(bool)
        out = self._run_on_device(lambda: self.backend.where(mask, value, self.data))
        return Tensor(out, dtype=self.dtype, device=self.device)

    def isinf(self):
        out = self._run_on_device(lambda: self.backend.isinf(self.data))
        return Tensor(out, dtype="bool", device=self.device)

    def clone(self) -> Tensor:
        out = self._run_on_device(lambda: self.data.copy())
        return Tensor(out, self.dtype, self.device)

    def clip(self, min_val: float, max_val: float):
        out = self._run_on_device(lambda: self.backend.clip(self.data, a_min=min_val, a_max=max_val))
        return Tensor(out, self.dtype, self.device)

    @staticmethod
    def pad(array, pad_width, mode="constant", dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, _, x: backend.pad(x, pad_width, mode=mode),
            array,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def concat(arr: list, axis: int = 0, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, _, *xs: backend.concatenate(xs, axis=axis),
            *arr,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def stack(arr: list, axis: int = 0, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, _, *xs: backend.stack(xs, axis=axis),
            *arr,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def zeros(shape, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, dt: backend.zeros(shape, dtype=dt),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def ones(shape, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, dt: backend.ones(shape, dtype=dt),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def eye(n: int, m: int | None = None, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, dt: backend.eye(n, m, dtype=dt),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def diag(array, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, _, x: backend.diag(x),
            array,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def where(condition, x, y, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, _, c, a, b: backend.where(c, a, b),
            condition,
            x,
            y,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def maximum(x1, x2, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, _, a, b: backend.maximum(a, b),
            x1,
            x2,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def minimum(x1, x2, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, _, a, b: backend.minimum(a, b),
            x1,
            x2,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def random_uniform(
        low: float = 0.0,
        high: float = 1.0,
        size: tuple = (2, 2),
        dtype="fp16",
        device="cpu",
    ) -> Tensor:
        return Tensor._static_op(
            lambda backend, dt: backend.random.uniform(low, high, size).astype(dt),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def random_normal(
        mean: float = 0.0,
        std: float = 1.0,
        size: tuple = (2, 2),
        dtype="fp16",
        device="cpu",
    ) -> Tensor:
        return Tensor._static_op(
            lambda backend, dt: backend.random.normal(mean, std, size).astype(dt),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def rand(shape: tuple, dtype="fp16", device="cpu") -> Tensor:
        return Tensor._static_op(
            lambda backend, dt: backend.random.rand(*shape).astype(dt),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def arange(
        start: float,
        stop: float | None = None,
        step: float = 1,
        dtype="fp32",
        device="cpu",
    ) -> Tensor:
        if stop is None:
            start, stop = 0, start

        return Tensor._static_op(
            lambda backend, dt: backend.arange(start, stop, step, dtype=dt),
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def einsum(
        subscripts: str,
        *operands,
        dtype="fp32",
        device="cpu",
    ) -> Tensor:
        return Tensor._static_op(
            lambda backend, _, *xs: backend.einsum(subscripts, *xs),
            *operands,
            dtype=dtype,
            device=device,
        )

    # ---------------------
    # Data type conversion methods
    # ---------------------

    def to_numpy(self):
        if self.backend.__name__ == "cupy":
            return self._data.get()
        else:
            return self._data

    def astype(self, dtype="fp16") -> Tensor:
        backend_dtype = Tensor.get_backend_dtype(self.backend, dtype)
        out = self._run_on_device(lambda: self._data.astype(backend_dtype, copy=False))
        return Tensor(out, dtype=dtype, device=self.device)

    # ---------------------
    # Representation
    # ---------------------

    def __repr__(self):
        return f"Tensor({self.data}, device={self.device}, dtype={self.dtype})"
