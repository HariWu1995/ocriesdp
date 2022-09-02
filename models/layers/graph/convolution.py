import math

import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.w_vi = nn.Parameter(torch.empty(in_dim, in_dim))
        self.w_vj = nn.Parameter(torch.empty(in_dim, in_dim))
        self.w_node = nn.Parameter(torch.empty(in_dim, out_dim))
        self.w_alpha = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias_h = nn.Parameter(torch.empty(in_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.w_vi, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_vj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_node, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w_alpha, a=math.sqrt(5))
        nn.init.uniform_(self.bias_h, a=0, b=1)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor, adj: torch.Tensor, box_num: torch.Tensor):
        """
        Parameters
        ----------
        x: nodes set (node embedding),    (B, N,    in_dim)
        adj: learned soft adj matrix,     (B, N, N)
        alpha: relation embedding,        (B, N, N, in_dim)
        box_num: (B, 1)

        Returns
        -------
        x_out: updated     node embedding, (B, N,    out_dim)
        alpha: updated relation embedding, (B, N, N, out_dim)
        """
        B, N, in_dim = x.shape

        # (B, N, N, in_dim)
        x_i = x.unsqueeze(2).expand(B, N, N, in_dim)
        x_j = x.unsqueeze(1).expand(B, N, N, in_dim)

        # (B, N, N, in_dim)
        x_i = torch.einsum('bijd, dk -> bijk', x_i, self.w_vi)
        x_j = torch.einsum('bijd, dk -> bijk', x_j, self.w_vj)

        # update hidden features between nodes, (B, N, N, in_dim）
        H = F.relu(x_i + x_j + alpha + self.bias_h)

        # update node embedding x, （B, N, out_dim）
        AH    = torch.einsum('bij, bijd -> bid', adj, H)
        new_x = torch.einsum('bid, dk   -> bik', AH, self.w_node)
        new_x = F.relu(new_x)

        # update relation embedding, (B, N, N, out_dim)
        new_alpha = torch.einsum('bijd, dk -> bijk', H, self.w_alpha)
        new_alpha = F.relu(new_alpha)

        return new_x, new_alpha


