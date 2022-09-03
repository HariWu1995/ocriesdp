from collections import namedtuple
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.activations import get_activation_fn


class Debug(nn.Module):
    
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name}')
        print(f'type: {type(x)}, len: {len(x)}')
        print(f'shapes: {self._get_shape_recurse(x)}')
        return x

    def _get_shape_recurse(self, x):
        if isinstance(x, torch.Tensor):
            return x.shape
        return [self._get_shape_recurse(a) for a in x]


class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """
    def __new__(cls, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)


class EmptyTensorOp(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return EmptyTensorOp.apply(grad, shape), None


class FeedForward(nn.Module):

    def __init__(self, input_dim: int, output_dim: Optional[int] = None,
                       hidden_dims: Optional[List[int]] = None, layer_norm: bool = False,
                       dropout: Optional[float] = None, activation: Optional[str] = 'relu'):
        super().__init__()
        self.input_dim = input_dim

        layers = []
        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(input_dim, dim))
                layers.append(get_activation_fn(activation))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout:
                    layers.append(nn.Dropout(dropout))
                input_dim = dim

        if not output_dim:
            self.output_dim = hidden_dims[-1]
            layers.append(nn.Identity())
        else:
            self.output_dim = output_dim
            layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor or List[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            x = torch.cat(x, 1)
        return self.mlp(x)


def Concat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)




