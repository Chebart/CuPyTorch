from ..blocks import GCNConv, ReLU
from .abstract_model import AbstractModel

class GCN(AbstractModel):  
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int
    ):
        self.layers = [
            GCNConv(in_features, hidden_features),
            ReLU(),
            GCNConv(hidden_features, out_features)
        ]
        super().__init__()

    def forward(self, x, edge_idx):
        for _, layer in enumerate(self.layers):
            if isinstance(layer, GCNConv):
                x = layer(x, edge_idx)
            else:
                x = layer(x)

        return x
    
    def backward(self, dLdy):
        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)