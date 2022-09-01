from typing import Any
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Sequential):

    def __init__(self, layer):
        # yapf: disable
        layers = [Parallel([layer, nn.Identity()]), Sum()]
        # yapf: enable
        super().__init__(*layers)


class ModulizedFunction(nn.Module):

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = partial(fn, *args, **kwargs)

    def forward(self, x):
        return self.fn(x)


class Interpolate(ModulizedFunction):

    def __init__(self, mode='bilinear', align_corners=False, **kwargs):
        super().__init__(
            F.interpolate, mode='bilinear', align_corners=False, **kwargs)


class SplitTensor(nn.Module):

    def __init__(self, size_or_sizes, dim):
        super().__init__()
        self.size_or_sizes = size_or_sizes
        self.dim = dim

    def forward(self, x):
        return x.split(self.size_or_sizes, dim=self.dim)


class Sum(nn.Module):

    def forward(self, x):
        return sum(x)


class AddAcross(nn.Module):

    def forward(self, x):
        return [sum(items) for items in zip(*x)]


class Reverse(nn.Module):

    def forward(self, x):
        return x[::-1]


class SelectOne(nn.Module):

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]


class Parallel(nn.ModuleList):
    ''' 
    Passes inputs through multiple `nn.Module`s in parallel.
    Returns a tuple of outputs.
    '''
    def forward(self, x: Any) -> tuple:
        # if multiple inputs, pass 
        #       1st input through 1st module,
        #       2nd input through 2nd module, and so on.
        if isinstance(x, (list, tuple)):
            return tuple(m(x) for m, x in zip(self, x))

        # if single input, pass it through all modules
        return tuple(m(x) for m in self)


class SequentialMultiOutput(nn.Sequential):
    """
    Like nn.Squential but returns all intermediate outputs as a tuple.
      input
        │
        │
        V
    [1st layer]───────> 1st out
        │
        │
        V
    [2nd layer]───────> 2nd out
        │
        │
        V
        .
        .
        .
        │
        │
        V
    [nth layer]───────> nth out
    """
    def forward(self, x: Any) -> tuple:
        outs = [None] * len(self)
        last_out = x
        for i, module in enumerate(self):
            last_out = module(last_out)
            outs[i] = last_out
        return tuple(outs)


class SequentialMultiInputMultiOutput(nn.Sequential):
    """
    Takes in either
    (1) an (n+1)-tuple of the form
      (last_out, 1st input, 2nd input, ..., nth input), or
    (2) an n-tuple of the form
      (1st input, 2nd input, ..., nth input),
    where n is the length of this sequential.

    If (2), the 1st layer in this sequential should be able to accept a single input. 
    All others are expected to accept a 2-tuple of inputs.
    
    Returns an n-tuple of all outputs of the form:
    (1st out, 2nd out, ..., nth out).
    
    In other words: the ith layer in this sequential takes in as inputs the
    ith input and the output of the last layer i.e. the (i-1)th layer.
    For the 1st layer, the "output of the last layer" is last_out.
    
                       last_out
                      (optional)
                          │
                          │
                          V
    1st input ───────[1st layer]───────> 1st out
                          │
                          │
                          V
    2nd input ───────[2nd layer]───────> 2nd out
                          │
                          │
                          V
        .                 .                  .
        .                 .                  .
        .                 .                  .
                          │
                          │
                          V
    nth input ───────[nth layer]───────> nth out
    """
    def forward(self, x: tuple) -> tuple:
        outs = [None] * len(self)

        if len(x) == len(self) + 1:
            last_out = x[0]
            layer_in = x[1:]
            layers = self
            start_idx = 0
        elif len(x) == len(self):
            last_out = self[0](x[0])
            layer_in = x[1:]
            layers = self[1:]
            outs[0] = last_out
            start_idx = 1
        else:
            raise ValueError('Invalid input format.')

        for i, (layer, x) in enumerate(zip(layers, layer_in), start_idx):
            last_out = layer((x, last_out))
            outs[i] = last_out

        return tuple(outs)

