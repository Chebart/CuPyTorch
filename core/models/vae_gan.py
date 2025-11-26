from ..blocks import Linear, Sigmoid, Conv2D, ConvTranspose2D, BatchNorm2d, ReLU, LeakyReLU, Reshape, Tanh
from .abstract_model import AbstractModel
from ..data import Tensor

class VAEEncoder(AbstractModel):
    def __init__(
        self,
        in_channels: int,
        z_dim: int
    ):
        self.z_dim = z_dim
        self.mu = 0
        self.sigma = 0
        self.layers = [
            Conv2D(in_channels, 32, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2d(32),
            LeakyReLU(0.2),
            Conv2D(32, 64, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2d(64),
            LeakyReLU(0.2),
            Conv2D(64, 128, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2d(128),
            LeakyReLU(0.2),
            Reshape(input_shape=(-1, 128, 4, 4), output_shape=(-1, 2048)),
            Linear(2048, 512),
            Linear(512, 2 * z_dim)
        ]
        super().__init__()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)

        self.mu = x[:, :self.z_dim]
        self.logvar = x[:, self.z_dim:]
        self.std = (0.5 * self.logvar).exp()
        self.eps = Tensor.random_normal(mean = 0, std = 1, size = self.mu.shape, dtype = self.mu.dtype, device = self.mu.device)
        return self.mu, self.logvar, self.eps * self.std + self.mu
    
    def backward(self, dLdy):
        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)   

class Generator(AbstractModel):
    def __init__(
        self,
        z_dim: int,
        out_channels: int
    ):
        self.layers = [
            Linear(z_dim, 512),
            Linear(512, 2048),
            Reshape(input_shape=(-1, 2048), output_shape=(-1, 128, 4, 4)),
            ConvTranspose2D(128, 64, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2d(64),
            ReLU(),
            ConvTranspose2D(64, 32, kernel_size = 4, stride = 2, padding = 1),
            BatchNorm2d(32),
            ReLU(),
            ConvTranspose2D(32, out_channels, kernel_size = 4, stride = 2, padding = 1),
            Tanh()
        ]
        super().__init__()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    def backward(self, dLdy):
        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)
        return dLdy

class Discriminator(AbstractModel):
    def __init__(
        self,
        in_channels: int,
        out_features: int
    ):
        self.layers = [
            Conv2D(in_channels, 32, kernel_size = 4, stride = 2, padding = 1),
            LeakyReLU(0.2),
            Conv2D(32, 64, kernel_size = 4, stride = 2, padding = 1),
            LeakyReLU(0.2),
            Conv2D(64, 128, kernel_size = 4, stride = 2, padding = 1),
            LeakyReLU(0.2),
            Reshape(input_shape=(-1, 128, 4, 4), output_shape=(-1, 2048)),
            Linear(2048, out_features),
            Sigmoid()
        ]
        super().__init__()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        return x
    
    def backward(self, dLdy):
        for _, layer in enumerate(reversed(self.layers)):
            dLdy = layer.backward(dLdy)
        return dLdy