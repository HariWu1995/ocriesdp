import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class SRNLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SRNLoss, self).__init__()
        self.loss_func = CrossEntropyLoss(reduction="sum")

    def forward(self, predicts, batch):
        predict = predicts['predict']
        word_predict = predicts['word_out']
        gsrm_predict = predicts['gsrm_out']
        label = batch[1]

        casted_label = label.int()
        casted_label = torch.reshape(casted_label, shape=[-1, 1])

        loss_word = self.loss_func(word_predict, label=casted_label)
        loss_gsrm = self.loss_func(gsrm_predict, label=casted_label)
        loss_vsfd = self.loss_func(     predict, label=casted_label)

        loss_word = torch.reshape(torch.sum(loss_word), shape=[1])
        loss_gsrm = torch.reshape(torch.sum(loss_gsrm), shape=[1])
        loss_vsfd = torch.reshape(torch.sum(loss_vsfd), shape=[1])

        loss_total = loss_word * 3.0 + loss_vsfd + loss_gsrm * 0.15

        return {
                  'loss': loss_total, 
             'word_loss': loss_word, 
            'image_loss': loss_vsfd,
        }


