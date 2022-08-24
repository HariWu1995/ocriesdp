import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class VQASerTokenLayoutLMLoss(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.loss_func = CrossEntropyLoss()
        self.num_classes = num_classes
        self.ignore_index = self.loss_func.ignore_index

    def forward(self, predicts, batch):
        labels, attention_mask = batch[1], batch[4]

        outputs = predicts.reshape([-1, self.num_classes])
        labels  =   labels.reshape([-1,                 ])
        if attention_mask is not None:
            attention_id = attention_mask.reshape([-1, ]) == 1
            outputs = outputs[attention_id]
            labels  =  labels[attention_id]
        loss = self.loss_func(outputs, labels)
        return loss

