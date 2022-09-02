import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    A layernorm module in the TF-style (epsilon inside the square root).
    """
    def __init__(self, d_model: int, variance: float=1e-11):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.variance = variance

    def forward(self, x: torch.Tensor ):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance)
        return self.gamma * x + self.beta


