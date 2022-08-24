import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import L1Loss, SmoothL1Loss, MSELoss as L2Loss, CrossEntropyLoss


EPS = 1e-7
INF = 1e11


def smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing: float=0.0):
    assert 0 <= smoothing < 1
    with torch.no_grad():
        targets = torch.empty(size=(targets.size(0), n_classes), device=targets.device) \
                        .fill_(smoothing / (n_classes-1)) \
                        .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
    return targets


class BCELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


class CELoss(nn.Module):

    def __init__(self, epsilon=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _label_smoothing(self, target, class_num):
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = smooth_one_hot(one_hot_target, smoothing=self.epsilon)
        soft_target = torch.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._label_smoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = torch.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(x, label=label, soft_label=soft_label)
        return loss


class KLJSLoss(object):
    """
    Kullback-Leibler and Jensen-Shannon Divergences
    """
    def __init__(self, mode: str = 'kl'):
        assert mode in ['kl', 'js', 'KL', 'JS'], "mode can only be 1 of ['kl', 'KL', 'js', 'JS']"
        self.mode = mode.lower()

    def __call__(self, p1, p2, reduction: str = "mean"):
        if self.mode == 'kl':
            loss = torch.multiply(p2, torch.log((p2 + EPS) / (p1 + EPS) + EPS))
            loss += torch.multiply(p1, torch.log((p1 + EPS) / (p2 + EPS) + EPS))
        elif self.mode == "js":
            loss = torch.multiply(p2, torch.log((2 * p2 + EPS) / (p1 + p2 + EPS) + EPS))
            loss += torch.multiply(p1, torch.log((2 * p1 + EPS) / (p1 + p2 + EPS) + EPS))
        else:
            raise ValueError("The mode.lower() if KLJSLoss should be one of ['kl', 'js']")

        loss *= 0.5
        if reduction == "mean":
            loss = torch.mean(loss, axis=[1, 2])
        elif reduction == "none" or reduction is None:
            return loss
        else:
            loss = torch.sum(loss, axis=[1, 2])

        return loss


class DMLLoss(nn.Module):
    """
    Distance-Metric-Learning Loss
    """
    def __init__(self, act=None, use_log=False):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None

        self.use_log = use_log
        self.loss_func = KLJSLoss(mode="kl")

    def _kldiv(self, x, target):
        loss = target * (torch.log(target + EPS) - x)
        # batch mean loss
        loss = torch.sum(loss) / loss.shape[0]
        return loss

    def forward(self, out1, out2):
        if self.act is not None:
            out1 = self.act(out1) + EPS
            out2 = self.act(out2) + EPS
        if self.use_log:
            # for recognition distillation, log is needed for feature map
            log_out1 = torch.log(out1)
            log_out2 = torch.log(out2)
            loss = (self._kldiv(log_out1, out2) + self._kldiv(log_out2, out1)) / 2.0
        else:
            # for detection distillation log is not needed
            loss = self.loss_func(out1, out2)
        return loss


class DistanceLoss(nn.Module):
    
    def __init__(self, mode="l2", **kargs):
        super().__init__()
        assert mode in ["l1", "l2", "smooth_l1"]
        if mode == "l1":
            self.loss_func = L1Loss(**kargs)
        elif mode == "l2":
            self.loss_func = L2Loss(**kargs)
        elif mode == "smooth_l1":
            self.loss_func = SmoothL1Loss(**kargs)

    def forward(self, x, y):
        return self.loss_func(x, y)


class ACELoss(nn.Module):
    """
    Aggregation Cross-Entropy Loss
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_func = CrossEntropyLoss(weight=None, ignore_index=0, reduction='none')

    def __call__(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]

        B, N = predicts.shape[:2]
        div = torch.tensor([N]).astype('float32')

        predicts = F.softmax(predicts, axis=-1)
        aggr_preds = torch.sum(predicts, axis=1)
        aggr_preds = torch.divide(aggr_preds, div)

        length = batch[2].astype("float32")
        batch = batch[3].astype("float32")
        batch[:, 0] = torch.subtract(div, length)
        batch = torch.divide(batch, div)

        loss = self.loss_func(aggr_preds, batch)
        return loss


class CenterLoss(nn.Module):
    """
    Reference: A Discriminative Feature Learning Approach for Deep Face Recognition
    """
    def __init__(self, num_classes: int=6625, feat_dim: int=96, center_file_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers = torch.randn(shape=[self.num_classes, self.feat_dim]).astype("float64")

        if center_file_path is not None:
            assert os.path.exists(center_file_path), \
                f"center path({center_file_path}) must exist when it is not None."
            with open(center_file_path, 'rb') as f:
                char_dict = pickle.load(f)
                for key in char_dict.keys():
                    self.centers[key] = torch.tensor(char_dict[key])

    def __call__(self, predicts, batch):
        assert isinstance(predicts, (list, tuple))
        features, predicts = predicts

        feats_reshape = torch.reshape(features, [-1, features.shape[-1]]).astype("float64")
        label = torch.argmax(predicts, axis=2)
        label = torch.reshape(label, [label.shape[0] * label.shape[1]])

        batch_size = feats_reshape.shape[0]

        #calc l2 distance between feats and centers  
        square_feat = torch.sum(torch.square(feats_reshape), axis=1, keepdim=True)
        square_feat = square_feat.expand([batch_size, self.num_classes])

        square_center = torch.sum(torch.square(self.centers), axis=1, keepdim=True)
        square_center = square_center.expand([self.num_classes, batch_size]).astype("float64")
        square_center = torch.transpose(square_center, [1, 0])

        distmat = torch.add(square_feat, square_center)
        feat_dot_center = torch.matmul(feats_reshape, torch.transpose(self.centers, [1, 0]))
        distmat = distmat - 2.0 * feat_dot_center

        #generate the mask
        classes = torch.arange(self.num_classes).astype("int64")
        label = torch.unsqueeze(label, 1).expand((batch_size, self.num_classes))
        mask = torch.equal(classes.expand([batch_size, self.num_classes]), label).astype("float64")
        dist = torch.multiply(distmat, mask)

        loss = torch.sum(torch.clip(dist, min=EPS, max=INF)) / batch_size
        return loss


class LossFromOutput(nn.Module):

    def __init__(self, key='loss', reduction='none'):
        super().__init__()
        self.key = key
        self.reduction = reduction

    def forward(self, predicts, batch):
        loss = predicts[self.key]
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss





