import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


EPS = 1e-7
INF = 1e7


class SDMGRLoss(nn.Module):

    def __init__(self, node_weight=1.0, edge_weight=1.0, ignore=0):
        super().__init__()
        self.loss_node = CrossEntropyLoss(ignore_index=ignore)
        self.loss_edge = CrossEntropyLoss(ignore_index=-1)
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.ignore = ignore

    def preprocess(self, gts, tag):
        gts, tag = gts.numpy(), tag.numpy().tolist()
        temp_gts = []
        batch = len(tag)
        for i in range(batch):
            num, recoder_len = tag[i][0], tag[i][1]
            temp_gts.append(torch.tensor(gts[i, :num, :num+1])).int()
        return temp_gts

    def accuracy(self, pred, target, topk=1, thresh=None):
        """
        Calculate accuracy according to the prediction and target.
        
        Args:
        -----
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

        Returns:
        --------
        float | tuple[float]: If ``topk`` is a integer, function will return a float as accuracy. 
            If ``topk`` is a tuple of multiple integers,  function will return a tuple of accuracies 
                                                            of each ``topk`` number.
        """
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
            return_single = True
        else:
            return_single = False

        maxk = max(topk)
        if pred.shape[0] == 0:
            accu = [torch.tensor(0.) for i in range(len(topk))]
            return accu[0] if return_single else accu

        pred_value, pred_label = torch.topk(pred, k=maxk, dim=1)
        pred_label = pred_label.transpose([1, 0])  # transpose to shape (maxk, N)
        correct = torch.equal(pred_label, (target.reshape([1, -1]).expand_as(pred_label)))
        res = []
        for k in topk:
            correct_k = torch.sum(correct[:k].reshape([-1]).float(), dim=0, keepdim=True)
            res.append(torch.multiply(correct_k, torch.tensor(100.0 / pred.shape[0])))

        return res[0] if return_single else res

    def forward(self, pred, batch):
        node_preds, edge_preds = pred
        gts, tag = batch[4], batch[5]
        gts = self.preprocess(gts, tag)
        node_gts, edge_gts = [], []
        for gt in gts:
            node_gts.append(gt[:, 0])
            edge_gts.append(gt[:, 1:].reshape([-1]))
        node_gts = torch.cat(node_gts)
        edge_gts = torch.cat(edge_gts)

        node_valids = torch.nonzero(node_gts != self.ignore).reshape([-1])
        edge_valids = torch.nonzero(edge_gts !=          -1).reshape([-1])
        loss_node = self.loss_node(node_preds, node_gts)
        loss_edge = self.loss_edge(edge_preds, edge_gts)
        loss = self.node_weight * loss_node + self.edge_weight * loss_edge
        return dict(
            loss=loss,
            loss_node=loss_node,
            loss_edge=loss_edge,
            acc_node=self.accuracy(torch.gather(node_preds, node_valids),
                                   torch.gather(node_gts, node_valids)),
            acc_edge=self.accuracy(torch.gather(edge_preds, edge_valids),
                                   torch.gather(edge_gts, edge_valids)),
        )