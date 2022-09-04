from copy import deepcopy

import torch
from torch import nn, distributed
from torch.nn import functional as F

from utils.comm.multi_gpu import get_world_size


EPS = 1e-7


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=EPS)
    x2 = (1 - x).clamp(min=EPS)
    return torch.log(x1 / x2)


def sigmoid_offset(x, offset=True):
    # modified sigmoid for range [-0.5, 1.5]
    if offset:
        return x.sigmoid() * 2 - 0.5
    else:
        return x.sigmoid()


def inverse_sigmoid_offset(x, offset=True):
    if offset:
        x = (x + 0.5) / 2.0
    return inverse_sigmoid(x)


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    A numerically-stable computation of logsumexp, used for summing log probabilities. 
    This is mathematically equivalent to `tensor.exp().sum(dim, keep=keepdim).log()`.

    Parameters
    ----------
    tensor : `torch.FloatTensor`, required.
        A tensor of arbitrary size.
    dim : `int`, optional (default = `-1`)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: `bool`, optional (default = `False`)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def reduce_sum(tensor: torch.Tensor):
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    tensor = tensor.clone()
    distributed.all_reduce(tensor, op=distributed.ReduceOp.SUM)
    return tensor


def reduce_mean(tensor: torch.Tensor):
    num_gpus = get_world_size()
    total = reduce_sum(tensor)
    return total.float() / num_gpus


def aligned_bilinear(tensor: torch.Tensor, factor):
    assert tensor.dim() == 4
    assert factor >= 1
    assert isinstance(factor, int)

    if factor == 1:
        return tensor

    h, w = tensor.size()[2:]
    tensor = F.pad(tensor, pad=(0, 1, 0, 1), mode="replicate")
    oh = factor * h + 1
    ow = factor * w + 1
    tensor = F.interpolate(tensor, size=(oh, ow), mode='bilinear', align_corners=True)
    tensor = F.pad(tensor, pad=(factor // 2, 0, 
                                factor // 2, 0), mode="replicate")

    return tensor[:, :, :oh-1, :ow-1]




