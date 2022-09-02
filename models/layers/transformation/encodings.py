"""
Original from https://github.com/tatp22/multidim-positional-encoding
"""
from typing import *
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.layers.common import LayerNorm


EPS = 1e-7
PI = math.pi


def get_positional_embeddings(input_tensor: torch.Tensor):
    """
    Get a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack([input_tensor.sin(), 
                       input_tensor.cos(),], dim=-1)
    return torch.flatten(emb, start_dim=-2, end_dim=-1)


class PositionalEncoding1D(nn.Module):

    def __init__(self, channels: int, temperature: float = 10_000., normalize: bool = False, scale: float = None):
        """
        channels: last dimension of the tensor to encode.
        """
        super(PositionalEncoding1D, self).__init__()
        channels = math.ceil(channels / 2) * 2
        self.channels = channels
        
        if normalize:
            if not isinstance(scale, float):
                scale = 2 * PI
        self.scale = scale
        self.normalize = normalize

        dim_t = torch.arange(0, self.channels, 2).float()
        inv_freq = 1.0 / (temperature ** (dim_t / self.channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cache = None

    def forward(self, tensor: torch.Tensor):
        """
        Params:
        -------
        tensor: A 3d tensor of size (B, P, C)

        Return: 
        -------
        Positional Encoding Matrix of size (B, P, C)
        """
        if len(tensor.shape) != 3:
            raise ValueError("Size of input tensor must be 3!")

        if self.cache is not None and self.cache.shape == tensor.shape:
            return self.cache

        self.cache = None
        B, P, C = tensor.shape

        pos_x = torch.arange(1, P+1, device=tensor.device).type(self.inv_freq.type())
        if self.normalize:
            pos_x = pos_x / (pos_x[-1:] + EPS) * self.scale

        inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_positional_embeddings(inp_x)
        emb = torch.zeros((P, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cache = emb[None, :, :C].repeat(B, 1, 1)
        return self.cache


class PositionalEncodingPermute1D(nn.Module):

    def __init__(self, channels: int, temperature: float = 10_000., normalize: bool = False, scale: float = None):
        """
        Get input of shape (B, C, P) instead of (B, P, C)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.pos_encoder = PositionalEncoding1D(channels, temperature, normalize, scale)

    def forward(self, tensor: torch.Tensor):
        tensor = tensor.permute(0, 2, 1)
        emb = self.pos_encoder(tensor)
        return emb.permute(0, 2, 1)

    @property
    def in_channels(self):
        return self.pos_encoder.in_channels

    @property
    def out_channels(self):
        return self.pos_encoder.out_channels


class PositionalEncoding2D(nn.Module):

    def __init__(self, channels: int, temperature: float = 10_000., normalize: bool = False, scale: float = None):
        """
        channels: last dimension of the tensor to encode.
        """
        super(PositionalEncoding2D, self).__init__()
        channels = math.ceil(channels / 4) * 2
        self.channels = channels

        if normalize:
            if not isinstance(scale, float):
                scale = 2 * PI
        self.scale = scale
        self.normalize = normalize

        dim_t = torch.arange(0, self.channels, 2).float()
        inv_freq = 1.0 / (temperature ** (dim_t / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("temperature", temperature)
        self.cache = None

    def forward(self, tensors):
        tensor = tensors.tensors
        mask = tensors.mask
        assert mask is not None, "Mask must be not null"

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + EPS) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + EPS) * self.scale

        dim_t = torch.arange(self.channels, dtype=torch.float32, device=tensor.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.channels)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_2d = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos_2d

    def forward_cache(self, tensor):
        """
        Params:
        -------
        tensor: A 4d tensor of size (B, Px, Py, C)

        Return: 
        -------
        Positional Encoding Matrix of size (B, Px, Py, C)
        """
        if len(tensor.shape) != 4:
            raise ValueError("Size of input tensor must be 4!")

        if self.cache is not None and self.cache.shape == tensor.shape:
            return self.cache

        self.cache = None
        B, Px, Py, C = tensor.shape
        pos_x = torch.arange(Px, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(Py, device=tensor.device).type(self.inv_freq.type())
        inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_positional_embeddings(inp_x).unsqueeze(1)
        emb_y = get_positional_embeddings(inp_y)
        emb = torch.zeros((Px, Py, self.channels * 2), device=tensor.device).type(tensor.type())
        emb[:,:,              :self.channels  ] = emb_x
        emb[:,:, self.channels:self.channels*2] = emb_y

        self.cache = emb[None, :, :, :C].repeat(B, 1, 1, 1)
        return self.cache


class PositionalEncodingPermute2D(nn.Module):

    def __init__(self, channels):
        """
        Get input of shape (B, C, Px, Py) instead of (B, Px, Py, C)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.pos_encoder = PositionalEncoding2D(channels)

    def forward(self, tensor: torch.Tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        emb = self.pos_encoder(tensor)
        return emb.permute(0, 3, 1, 2)

    @property
    def in_channels(self):
        return self.pos_encoder.in_channels

    @property
    def out_channels(self):
        return self.pos_encoder.out_channels


class PositionalEncoding3D(nn.Module):

    def __init__(self, channels: int, temperature: float = 10_000.):
        """
        channels: last dimension of the tensor to encode.
        """
        super(PositionalEncoding3D, self).__init__()
        self.in_channels = channels
        channels = math.ceil(channels / 6) * 2
        if channels % 2:
            channels += 1
        self.out_channels = channels
        inv_freq = 1.0 / (temperature ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cache = None

    def forward(self, tensor):
        """
        Params:
        -------
        tensor: A 5d tensor of size (B, Px, Py, Pz, C)

        Return: 
        -------
        Positional Encoding Matrix of size (B, Px, Py, Pz, C)
        """
        if len(tensor.shape) != 5:
            raise ValueError("Size of input tensor must be 5!")

        if self.cache is not None and self.cache.shape == tensor.shape:
            return self.cache

        self.cache = None
        B, Px, Py, Pz, C = tensor.shape
        
        pos_x = torch.arange(Px, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(Py, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(Pz, device=tensor.device).type(self.inv_freq.type())
        
        inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)

        emb_x = get_positional_embeddings(inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_positional_embeddings(inp_y).unsqueeze(1)
        emb_z = get_positional_embeddings(inp_z)
        emb = torch.zeros((Px, Py, Pz, self.channels * 3), device=tensor.device).type(tensor.type())
        emb[:,:,:,                 :   self.channels] = emb_x
        emb[:,:,:,   self.channels : 2*self.channels] = emb_y
        emb[:,:,:, 2*self.channels :                ] = emb_z

        self.cache = emb[None, :, :, :, :C].repeat(B, 1, 1, 1, 1)
        return self.cache


class PositionalEncodingPermute3D(nn.Module):

    def __init__(self, channels):
        """
        Get input of shape (B, C, Px, Py, Pz) instead of (B, Px, Py, Pz, C)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.pos_encoder = PositionalEncoding3D(channels)

    def forward(self, tensor: torch.Tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        emb = self.pos_encoder(tensor)
        return emb.permute(0, 4, 1, 2, 3)

    @property
    def in_channels(self):
        return self.pos_encoder.in_channels

    @property
    def out_channels(self):
        return self.pos_encoder.out_channels
 

class PositionalEncodingLearnable(nn.Module):

    def __init__(self, d_model, dropout: float=0.1, max_seq_len: int=100):
        super(PositionalEncodingLearnable, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(self.layernorm(x))


class PositionalEncodingAggregator(nn.Module):

    def __init__(self, pos_encoder):
        super(PositionalEncodingAggregator, self).__init__()
        self.pos_encoder = pos_encoder

    def forward(self, tensor):
        """
        :param tensor: A 3, 4 or 5d tensor that matches the model output size
        :return: Positional Encoding Matrix summed to the original tensor
        """
        emb = self.pos_encoder(tensor)
        assert emb.size() == tensor.size(), f"Size of input tensor ({tensor.size()}) mismathces size of positional encoder {emb.size()}"
        return emb + tensor


