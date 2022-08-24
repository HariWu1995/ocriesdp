import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


EPS = 1e-7
INF = 1e7


class ClsLoss(nn.Module):

    def __init__(self, **kwargs):
        super(ClsLoss, self).__init__()
        self.loss_func = CrossEntropyLoss(reduction='mean')

    def forward(self, predicts, batch):
        label = batch[1].astype("int64")
        loss = self.loss_func(input=predicts, label=label)
        return loss


class AttentionLoss(nn.Module):

    def __init__(self, **kwargs):
        super(AttentionLoss, self).__init__()
        self.loss_func = CrossEntropyLoss(weight=None, reduction='none')

    def forward(self, predicts, batch):
        targets       = batch[1].astype("int64")
        label_lengths = batch[2].astype('int64')
        batch_size, num_steps, num_classes = predicts.shape[0], predicts.shape[1], predicts.shape[2]
        assert len(targets.shape) == len(list(predicts.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = torch.reshape(predicts, [-1, predicts.shape[-1]])
        targets = torch.reshape(targets, [-1])

        return torch.sum(self.loss_func(inputs, targets))


class MultiLoss(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.loss_list = kwargs.pop('loss_config_list')
        self.weight_1 = kwargs.get('weight_1', 1.0)
        self.weight_2 = kwargs.get('weight_2', 1.0)
        self.gtc_loss = kwargs.get('gtc_loss', 'sar')

        self.loss_funcs = {}
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def forward(self, predicts, batch):
        # batch [image, label_ctc, label_sar, length, valid_ratio]

        self.total_loss = {}
        total_loss = 0.0
        for name, loss_func in self.loss_funcs.items():
            if name == 'CTCLoss':
                loss = loss_func(predicts['ctc'], batch[:2] + batch[3:])['loss'] * self.weight_1
            elif name == 'SARLoss':
                loss = loss_func(predicts['sar'], batch[:1] + batch[2:])['loss'] * self.weight_2
            else:
                raise NotImplementedError(
                    '{} is not supported in MultiLoss yet'.format(name))
            self.total_loss[name] = loss
            total_loss += loss
        self.total_loss['loss'] = total_loss
        return self.total_loss

