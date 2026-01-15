from abc import ABC, abstractmethod

from core.data import Tensor

class AbstractBlock(ABC):
    def __call__(self, x: Tensor)-> Tensor:
        return self.forward(x)

    @abstractmethod
    def forward(self, x: Tensor)-> Tensor:
        """Calculate forward pass"""
        pass

    @abstractmethod
    def parameters(self)-> list[tuple[str, Tensor, Tensor]]:
        """Return parameters of the block"""
        pass

    def to_device(self, device: str):
        """Move all model blocks to needed device"""
        for _, p, g in self.parameters():
            p = p.to_device(device)
            g = g.to_device(device)

    def train(self):
        """Prepare block for training"""
        self.is_train = True

    def eval(self):
        """Prepare block for evaluation"""
        self.is_train = False

    @abstractmethod
    def backward(self, dLdy: Tensor)-> Tensor:
        """Calculate backward pass"""
        pass