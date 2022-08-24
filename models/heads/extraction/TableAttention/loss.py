import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


EPS = 1e-7
INF = 1e11


class TableAttentionLoss(nn.Module):

    def __init__(self, structure_weight, loc_weight, use_giou=False, giou_weight=1.0, **kwargs):
        super(TableAttentionLoss, self).__init__()
        self.loss_func = CrossEntropyLoss(weight=None, reduction='none')
        self.structure_weight = structure_weight
        self.loc_weight = loc_weight
        self.giou_weight = giou_weight
        self.use_giou = use_giou
        
    def giou_loss(self, preds, bbox, eps=1e-7, reduction='mean'):
        '''
        :param
        preds  [[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        bbox   [[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
        
        :return: 
        loss
        '''
        ix1 = torch.maximum(preds[:, 0], bbox[:, 0])
        iy1 = torch.maximum(preds[:, 1], bbox[:, 1])
        ix2 = torch.minimum(preds[:, 2], bbox[:, 2])
        iy2 = torch.minimum(preds[:, 3], bbox[:, 3])
        iw = torch.clip(ix2 - ix1 + EPS, 0., INF)
        ih = torch.clip(iy2 - iy1 + EPS, 0., INF)

        # overlap
        inters = iw * ih

        # union
        uni = (preds[:, 2] - preds[:, 0] + EPS) * (preds[:, 3] - preds[:, 1] + EPS) \
            + ( bbox[:, 2] -  bbox[:, 0] + EPS) * ( bbox[:, 3] -  bbox[:, 1] + EPS) - inters + EPS

        # ious
        ious = inters / uni

        ex1 = torch.minimum(preds[:, 0], bbox[:, 0])
        ey1 = torch.minimum(preds[:, 1], bbox[:, 1])
        ex2 = torch.maximum(preds[:, 2], bbox[:, 2])
        ey2 = torch.maximum(preds[:, 3], bbox[:, 3])
        ew = torch.clip(ex2 - ex1 + EPS, 0., INF)
        eh = torch.clip(ey2 - ey1 + EPS, 0., INF)

        # enclose erea
        enclose = ew * eh + eps
        giou = ious - (enclose - uni) / enclose

        loss = 1 - giou

        if reduction == 'mean':
            loss = torch.mean(loss)
        elif reduction == 'sum':
            loss = torch.sum(loss)
        return loss

    def forward(self, predicts, batch):
        structure_probs = predicts['structure_probs']
        structure_targets = batch[1].to(torch.int64)
        structure_targets = structure_targets[:, 1:]
        if len(batch) == 6:
            structure_mask = batch[5].to(torch.int64)
            structure_mask = structure_mask[:, 1:]
            structure_mask = torch.reshape(structure_mask, [-1])
        structure_probs = torch.reshape(structure_probs, [-1, structure_probs.shape[-1]])
        structure_targets = torch.reshape(structure_targets, [-1])
        structure_loss = self.loss_func(structure_probs, structure_targets)
        
        if len(batch) == 6:
             structure_loss = structure_loss * structure_mask
        structure_loss = torch.mean(structure_loss) * self.structure_weight
        
        loc_preds = predicts['loc_preds']
        loc_targets      = batch[2].to(torch.float32)
        loc_targets_mask = batch[4].to(torch.float32)
        loc_targets      = loc_targets[     :, 1:, :]
        loc_targets_mask = loc_targets_mask[:, 1:, :]
        loc_loss = F.mse_loss(loc_preds * loc_targets_mask, loc_targets) * self.loc_weight

        loss_dict = dict(structure_loss=structure_loss, loc_loss=loc_loss,)
        if self.use_giou:
            loc_loss_giou = self.giou_loss(loc_preds * loc_targets_mask, loc_targets) * self.giou_weight
            total_loss = structure_loss + loc_loss + loc_loss_giou
            loss_dict.update(dict(loss=total_loss, loc_loss_giou=loc_loss_giou,))
        else:
            total_loss = structure_loss + loc_loss            
            loss_dict.update(dict(loss=total_loss,))
        return loss_dict
