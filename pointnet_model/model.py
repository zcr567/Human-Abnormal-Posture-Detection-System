import torch
from torch import save, max
from torch.nn import Linear, ReLU, Module, Sequential, Conv2d, MaxPool1d, Flatten, ModuleList

point_amount = 4096


# TODO: make a more detailed test

class TNet(Module):
    """Defines a T-Net, which return a transformation matrix to apply to the input data."""

    def __init__(self, size):
        """:param size: length of the input data."""
        super(TNet, self).__init__()
        self.size = size
        self.model = Sequential(
            Flatten(0, 1),
            Linear(point_amount * size, size ** 2),
        )

    def forward(self, x):
        flatten = self.model(x)
        mat = flatten.view(self.size, self.size)
        return mat


class MLP(Module):
    """Defines a MLP."""

    def __init__(self, shape: tuple):
        """:param shape: Shape of the input data. Features of each layers, from input to output."""
        super(MLP, self).__init__()
        layers = ModuleList()
        for i in range(len(shape) - 1):
            layers.append(Linear(shape[i], shape[i + 1]))
            layers.append(ReLU())
        layers.pop(-1)  # pop the last RelU layer

        self.model = Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PointNet(Module):

    def __init__(self):
        super(PointNet, self).__init__()
        self.input_t = TNet(3)
        self.mlp1 = MLP((3, 64, 64))
        self.feature_t = TNet(64)
        self.mlp2 = MLP((64, 64, 128, 1024))
        # self.maxpool = MaxPool1d(point_amount)
        self.maxpool = max
        self.mlp3 = MLP((1024, 512, 256, 2))

    def forward(self, x):
        x @= self.input_t(x)
        x = self.mlp1(x)
        x @= self.feature_t(x)
        x = self.mlp2(x)
        x = self.maxpool(x, 0)[0]
        x = self.mlp3(x)
        return x

    def save(self, f='pointnet.pt'):
        save(self.state_dict(), f)


if __name__ == '__main__':
    with torch.no_grad:
        # A basic test to ensure that the programming is right
        module = PointNet()
        inpt = torch.randn(4096, 3)
        print(inpt.shape)
        print(module(inpt))
