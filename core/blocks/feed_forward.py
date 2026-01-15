from .abstract_block import AbstractBlock
from .linear import Linear
from .relu import ReLU

class FeedForwardLayer(AbstractBlock):
    """FeedForwardLayer(x) = linear(relu(linear(x)))"""

    def __init__(
        self, 
        d_model: int, 
        d_ff: int
    ):
        # Initialize needed components
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.relu = ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

    def backward(self, dLdy):
        return self.fc1.backward(self.relu.backward(self.fc2.backward(dLdy)))