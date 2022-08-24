import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class SARLoss(nn.Module):

    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        ignore_index = kwargs.get('ignore_index', 92)  # 6626
        self.loss_func = CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def forward(self, predicts, batch):
        predict = predicts[:, :-1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = batch[1].astype("int64")[:, 1:]  # ignore first index of target in loss calculation
        batch_size, num_steps, num_classes = predict.shape[:3]
        assert len(label.shape) == len(list(predict.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = torch.reshape(predict, [-1, num_classes])
        targets = torch.reshape(label, [-1])
        return self.loss_func(inputs, targets)


