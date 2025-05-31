import torch as tf
import torch.nn as nn

from typing import Union


class MyReshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()

        self.shape = shape

    def forward(self, x):
        x = x.view(*self.shape)

        return x

class MyPermute(nn.Module):
    def __init__(self, *order):
        super().__init__()

        self.order = order

    def forward(self, x):
        x = x.permute(*self.order)

        return x