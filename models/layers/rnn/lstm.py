from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.common import FeedForward


class BidirLSTMClassification(nn.Module):

    def __init__(self, lstm_config, mlp_config, padding_value: int = 0):
        super().__init__()
        self.padding_value = padding_value  # keys_vocab_cls.stoi['<pad>']
        self.lstm = nn.LSTM(**lstm_config)
        self.ff = FeedForward(**mlp_config)

    @staticmethod
    def sort_tensor(x: torch.Tensor, length: torch.Tensor, h_0: torch.Tensor = None, c_0: torch.Tensor = None):
        sorted_lenght, sorted_order = torch.sort(length, descending=True)
        _, invert_order = sorted_order.sort(0, descending=False)
        if h_0 is not None:
            h_0 = h_0[:, sorted_order, :]
        if c_0 is not None:
            c_0 = c_0[:, sorted_order, :]
        return x[sorted_order], sorted_lenght, invert_order, h_0, c_0

    def forward(self, x_seq: torch.Tensor, lenghts: torch.Tensor, initial: Tuple[torch.Tensor, torch.Tensor]):
        """
        Parameters
        ----------
        x_seq: (B, N*T, D)
        lenghts: (B,)
        initial: (num_layers * num_directions, B, D)
        
        Returns
        -------
        logits: (B, N*T, out_dim)
        """
        # B*N, T, hidden_size
        x_seq, sorted_lengths, invert_order, h_0, c_0 = self.sort_tensor(x_seq, lenghts, initial[0], initial[0])
        packed_x = nn.utils.rnn.pack_padded_sequence(x_seq, batch_first=True, lengths=sorted_lengths)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True, padding_value=self.padding_value)

        # total_length = MAX_BOXES_NUM * MAX_TRANSCRIPT_LEN
        output = output[invert_order]
        logits = self.ff(output) # (B, N*T, out_dim)
        return logits


