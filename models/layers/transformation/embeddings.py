from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models.activations import get_activation_fn
from models.layers.common import LayerNorm


class Embeddings2D(nn.Module):

    def __init__(self, config):
        super(Embeddings2D, self).__init__()
        self.token_id_embeddings     = nn.Embedding(config.id_vocab_size,              config.hidden_size, padding_idx=0)
        self.token_type_embeddings   = nn.Embedding(config.type_vocab_size,            config.hidden_size, )
        self.token_size_embeddings   = nn.Embedding(config.token_max_size,             config.hidden_size, )
        self.token_dir_embeddings    = nn.Embedding(config.token_num_direction,        config.hidden_size, )    # 2: horizontal + vertical
        self.position1D_embeddings   = nn.Embedding(config.max_1d_position_embeddings, config.hidden_size, )
        self.position2D_x_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size, )
        self.position2D_y_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size, )
        self.position2D_h_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size, )
        self.position2D_w_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.hidden_size, )

        # LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load any TF checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_ids, bbox, token_type_ids=None, position_ids=None, inputs_embeds=None, 
                                       token_bbox_size=None, token_direction=None,):
        seq_length = token_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=token_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(token_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(token_ids)
        if token_bbox_size is None:
            token_bbox_size = torch.zeros_like(token_ids)
        if token_direction is None:
            token_direction = torch.zeros_like(token_ids)

        # Embeddings for token information
        token_embeddings = self.token_id_embeddings(token_ids) + \
                         self.token_type_embeddings(token_type_ids) + \
                         self.token_size_embeddings(token_bbox_size) + \
                          self.token_dir_embeddings(token_direction)

        # Embeddings for position (1D and 2D) information
        position_embeddings        = self.position1D_embeddings(position_ids)
        position_embeddings_left   = self.position2D_x_embeddings(bbox[:, :, 0])
        position_embeddings_upper  = self.position2D_y_embeddings(bbox[:, :, 1])
        position_embeddings_right  = self.position2D_x_embeddings(bbox[:, :, 2])
        position_embeddings_lower  = self.position2D_y_embeddings(bbox[:, :, 3])
        position_embeddings_height = self.position2D_h_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
        position_embeddings_width  = self.position2D_w_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
        position_embeddings += position_embeddings_left   + position_embeddings_upper + \
                               position_embeddings_right  + position_embeddings_lower + \
                               position_embeddings_height + position_embeddings_width

        # Aggregation
        embeddings = token_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class BertEncoder2D(pl.LightningModule):
    """
    The variable names from transformers.models.bert.modeling_bert.BertEncoder
    Other classes also follow the same convention to load the pretrained weights.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayer2D(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states, rn_emb, attention_mask):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, rn_emb, attention_mask)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
