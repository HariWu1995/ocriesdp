from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.graph.convolution import GraphConvolution


MAX_BOXES_NUM = 70  # max number of boxes per document
MAX_SEQ_LEN = 50  # max length of text per box


class GraphLearning(nn.Module):

    def __init__(self, in_dim: int, learning_dim: int, gamma: float, eta: float):
        super().__init__()
        self.projection = nn.Linear(in_dim, learning_dim, bias=False)
        self.adjacency = nn.Parameter(torch.empty(learning_dim))
        self.gamma = gamma
        self.eta = eta
        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.adjacency, a=0, b=1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, box_num: torch.Tensor = None):
        """
        Parameters
        ----------
        x: nodes set, (B*N, D)
        adj: adjacency matrix, (B, N, N, default is 1)
        box_num: (B, 1)
        
        Returns
        -------
        soft_adj_matrix
        gl_loss
        """
        B, N, D = x.shape 
        
        # (B, N, D)
        x_hat = self.projection(x)
        _, _, learning_dim = x_hat.shape

        # (B, N, N, learning_dim)
        x_i = x_hat.unsqueeze(2).expand(B, N, N, learning_dim)
        x_j = x_hat.unsqueeze(1).expand(B, N, N, learning_dim)

        # (B, N, N, learning_dim)
        distance = torch.abs(x_i - x_j)

        # add -1 flag to distance, if node is not exist. to separate normal node distances from not exist node distance.
        if box_num is not None:
            # mask = self.compute_static_mask(box_num)
            mask = self.compute_dynamic_mask(box_num)
            distance = distance + mask

        # (B, N, N)
        distance = torch.einsum('bijd, d -> bij', distance, self.adjacency)
        out = F.leaky_relu(distance)

        # for numerical stability, due to softmax operation producing large value
        max_out_v, _ = out.max(dim=-1, keepdim=True)
        out = out - max_out_v

        soft_adj = torch.exp(out)
        soft_adj = adj * soft_adj

        sum_out = soft_adj.sum(dim=-1, keepdim=True)
        soft_adj = soft_adj / sum_out + 1e-10

        gl_loss = None
        if self.training:
            gl_loss = self._graph_learning_loss(x_hat, soft_adj, box_num)

        return soft_adj, gl_loss

    @staticmethod
    def compute_static_mask(box_num: torch.Tensor):
        """
        Compute -1 mask, if node(box) is not exist, the length of mask is MAX_BOXES_NUM,
        this will help with single-node multi-gpu training mechanism, and ensure batch shape is same. 
        But this operation leads to waste memory.

        Parameters
        ----------
        box_num: (B, 1)
        
        Returns
        -------
        mask: (B, N, N, 1)
        """
        max_len = MAX_BOXES_NUM

        # (B, N)
        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))
        box_num = box_num.expand_as(mask)
        mask = mask < box_num

        row_mask = mask.unsqueeze(1) # (B, 1, N)
        col_mask = mask.unsqueeze(2) # (B, N, 1)        
        mask = (row_mask & col_mask) # (B, N, N)

        # -1 if not exist node, or 0
        mask = ~mask * -1

        return mask.unsqueeze(-1)

    @staticmethod
    def compute_dynamic_mask(box_num: torch.Tensor):
        """
        Compute -1 mask, if node(box) is not exist, the length of mask is calculate by max(box_num),
        this will help with multi-nodes multi-gpu training mechanism, ensure batch of different gpus have same shape.

        Parameters
        ----------
        box_num: (B, 1)
        
        Returns
        -------
        mask: (B, N, N, 1)
        """
        max_len = torch.max(box_num)

        # (B, N)
        mask = torch.arange(0, max_len, device=box_num.device).expand((box_num.shape[0], max_len))
        box_num = box_num.expand_as(mask) 
        mask = mask < box_num

        row_mask = mask.unsqueeze(1) # (B, 1, N)
        col_mask = mask.unsqueeze(2) # (B, N, 1)
        mask = (row_mask & col_mask) # (B, N, N)

        # -1 if not exist node, or 0
        mask = ~mask * -1

        return mask.unsqueeze(-1)

    def _graph_learning_loss(self, x_hat: torch.Tensor, adj: torch.Tensor, box_num: torch.Tensor):
        """
        Calculate graph learning loss
        
        Parameters
        ----------
        x_hat: (B, N, D)
        adj: (B, N, N)
        box_num: (B, 1)

        Returns
        -------
        gl_loss
        """
        B, N, D = x_hat.shape
        
        # (B, N, N, out_dim)
        x_i = x_hat.unsqueeze(2).expand(B, N, N, D)
        x_j = x_hat.unsqueeze(1).expand(B, N, N, D)

        box_num_div = 1 / torch.pow(box_num.float(), 2) # (B, 1)
        
        dist_loss = adj + self.eta * torch.norm(x_i - x_j, dim=3) # remove square operation due to nan
        dist_loss = torch.exp(dist_loss) # (B, N, N)
        dist_loss = torch.sum(dist_loss, dim=(1, 2)) * box_num_div.squeeze(-1)  # (B,)
        f_norm = torch.norm(adj, dim=(1, 2)) # remove square operation due to nan
        gl_loss = dist_loss + self.gamma * f_norm
        return gl_loss


class GraphLearningConvolutionNetwork(nn.Module):
    """
    Paper: https://arxiv.org/pdf/1811.09971.pdf
    """
    def __init__(self, in_dim: int, out_dim: int, gamma: float = 0.0001, eta: float = 1,
                 learning_dim: int = 128, num_layers: int =2):
        super().__init__()

        self.adjacency = GraphLearning(in_dim=in_dim, gamma=gamma, eta=eta, learning_dim=learning_dim)
        self.transform = nn.Linear(6, in_dim, bias=False)

        convols = []
        in_dim_i = in_dim
        for i in range(num_layers):
            convols.append(GraphConvolution(in_dim_i, out_dim))
            in_dim_i = out_dim
            out_dim = in_dim_i
        self.convols = nn.ModuleList(convols)
        
    def forward(self, x: torch.Tensor, relations: torch.Tensor, adj: torch.Tensor, box_num: torch.Tensor, **kwargs):
        """
        Parameters
        ----------
        nodes      :     nodes embedding, (B*N, D)
        relations  : relations embedding, (B, N, N, 6)
        adj: adjacent matrix, (B, N, N)
        box_num: (B, 1)
        """
        # relation features embedding, (B, N, N, in_dim)
        alpha = self.transform(relations)

        soft_adj, gl_loss = self.adjacency(x, adj, box_num)
        adj = adj * soft_adj
        for i, gcn_layer in enumerate(self.convols):
            x, alpha = gcn_layer(x, alpha, adj, box_num)

        return x, soft_adj, gl_loss



