from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F


def max_by_axis(tensors):
    # type: (List[List[int]]) -> List[int]
    maxes = tensors[0]
    for tensor in tensors[1:]:
        for index, item in enumerate(tensor):
            maxes[index] = max(maxes[index], item)
    return maxes


def batchify_tensors(tensor_list: List[torch.Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        max_size = max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError(f'tensor must be 3D, instead {tensor_list[0].ndim}')
    return BatchedTensor(tensor, mask)


class BatchedTensor(object):

    def __init__(self, tensors: torch.Tensor, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        tensors = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            mask = mask.to(device)
        return BatchedTensor(tensors, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)




