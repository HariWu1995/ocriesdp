import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class NRTRLoss(nn.Module):

    def __init__(self, smoothing=True, **kwargs):
        super(NRTRLoss, self).__init__()
        self.loss_func = CrossEntropyLoss(reduction='mean', ignore_index=0)
        self.smoothing = smoothing

    def forward(self, pred, batch):
        pred = pred.reshape([-1, pred.shape[2]])
        max_len = batch[2].max()
        tgt = batch[1][:, 1:2 + max_len]
        tgt = tgt.reshape([-1])
        if self.smoothing:
            eps = 0.1
            n_classes = pred.shape[1]
            one_hot = F.one_hot(tgt, pred.shape[1])
            one_hot = one_hot * (1-eps) + (1 - one_hot) * eps / (n_classes-1)
            log_prb = F.log_softmax(pred, dim=1)
            non_pad_mask = torch.not_equal(tgt, torch.zeros(tgt.shape, dtype=tgt.dtype))
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).mean()
        else:
            loss = self.loss_func(pred, tgt)
        return loss