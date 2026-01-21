from abc import ABC, abstractmethod

from core.data import Tensor

class AbstractModel(ABC):
    def __init__(self):
        if not hasattr(self, "layers"):
            raise NotImplementedError(
                "Models must have self.layers attribute!"
            )
        self.device = "cpu"

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> Tensor:
        """Calculate forward pass"""
        pass

    def parameters(self)-> list[tuple[str, Tensor, Tensor]]:
        """Return parameters of the model"""
        params = []
        for layer in self.layers:
            for p_type, p, g in layer.parameters():
                params.append((p_type, p, g))
        return params

    def to_device(self, device: str):
        """Move all model blocks to needed device"""
        self.device = device
        for layer in self.layers:
            layer.to_device(device)

        return self

    @abstractmethod
    def backward(self, dLdy: Tensor)-> None:
        """Calculate backward pass"""
        pass