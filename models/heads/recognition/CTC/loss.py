import torch
from torch import nn

from src.models.losses.common import ACELoss, CenterLoss


class CTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLoss, self).__init__()
        self.loss_func = CTCLoss(blank=0, reduction='none')
        self.use_focal_loss = use_focal_loss

    def forward(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = torch.tensor([N] * B, dtype='int64')
        label_lengths = batch[2].astype('int64')
        labels        = batch[1].astype("int32")
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = torch.exp(-loss)
            weight = torch.subtract(torch.tensor([1.0]), weight)
            weight = torch.square(weight)
            loss = torch.multiply(loss, weight)

        return loss.mean()


class EnhancedCTCLoss(nn.Module):

    def __init__(self, use_focal_loss=False, num_classes=6625, feat_dim=96, 
                        use_ace_loss=False, ace_loss_weight=0.1,
                        use_center_loss=False, center_loss_weight=0.05, init_center=False, center_file_path=None, **kwargs):
        super(EnhancedCTCLoss, self).__init__()
        self.ctc_loss_func = CTCLoss(use_focal_loss=use_focal_loss)

        self.use_ace_loss = False
        if use_ace_loss:
            self.use_ace_loss = use_ace_loss
            self.ace_loss_func = ACELoss()
            self.ace_loss_weight = ace_loss_weight

        self.use_center_loss = False
        if use_center_loss:
            self.use_center_loss = use_center_loss
            self.center_loss_func = CenterLoss(num_classes=num_classes, feat_dim=feat_dim,
                                                init_center=init_center, center_file_path=center_file_path)
            self.center_loss_weight = center_loss_weight

    def __call__(self, predicts, batch):
        loss = self.ctc_loss_func(predicts, batch)["loss"]

        if self.use_center_loss:
            center_loss = self.center_loss_func(predicts, batch)["loss_center"] 
            loss = loss + center_loss * self.center_loss_weight

        if self.use_ace_loss:
            ace_loss = self.ace_loss_func(predicts, batch)["loss_ace"]
            loss = loss + ace_loss * self.ace_loss_weight

        return loss

