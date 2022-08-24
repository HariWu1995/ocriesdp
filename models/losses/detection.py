import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, MSELoss


EPS = 1e-7
INF = 1e7


class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, gt, mask, weights=None):
        assert pred.shape == gt.shape, f"Expect shape of predictions ({pred.size()}) equal to shape of labels ({gt.size()})"
        assert pred.shape == mask.shape, f"Expect shape of predictions ({pred.size()}) equal to shape of masks ({mask.size()})"
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = torch.sum(pred * gt * mask)

        union = torch.sum(pred * mask) + torch.sum(gt * mask) + EPS
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskedL1Loss(nn.Module):

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + EPS)
        loss = torch.mean(loss)
        return loss


class BalanceLoss(nn.Module):
    """
    The BalanceLoss for Differentiable Binarization text detection
    
    Params:
    -------
    balance_loss (bool): whether balance loss or not
        default is True
    loss_type (str): 1 of ['CrossEntropy','DiceLoss', 'Euclidean','BCELoss', 'MaskL1Loss']
        default is  'DiceLoss'.
    negative_ratio (int|float): float
        default is 3.
    return_origin (bool): whether return unbalanced loss or not
        default is False.
    """
    valid_losses = ['CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss', 'MaskL1Loss']

    def __init__(self, balance_loss=True, loss_type='DiceLoss', negative_ratio=3, return_origin=False, **kwargs):
       
        super(BalanceLoss, self).__init__()
        self.loss_type = loss_type
        self.balance_loss = balance_loss
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin

        if self.loss_type == "CrossEntropy":
            self.loss_func = CrossEntropyLoss()
        elif self.loss_type == "Euclidean":
            self.loss_func = MSELoss()
        elif self.loss_type == "DiceLoss":
            self.loss_func = DiceLoss()
        elif self.loss_type == "BCELoss":
            self.loss_func = BCELoss(reduction='none')
        elif self.loss_type == "MaskL1Loss":
            self.loss_func = MaskedL1Loss()
        else:
            raise Exception(f"loss_type in BalanceLoss() can only be one of {self.valid_losses}")

    def forward(self, pred, gt, mask=None):
        positive =      gt  * mask
        negative = (1 - gt) * mask

        positive_count = int(    positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss_func(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            # negative_loss, _ = torch.topk(negative_loss, k=negative_count_int)
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                            positive_count      + negative_count + EPS)
        else:
            balance_loss = positive_loss.sum() / (positive_count + EPS)

        if self.return_origin:
            return balance_loss, loss
        return balance_loss



