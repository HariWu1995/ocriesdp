import logging

import torch
from torch import nn

from transformers import BertConfig, BertModel, BertPreTrainedModel

from models.layers import Embeddings2D

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP = {}
LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class LayoutlmConfig(BertConfig):

    pretrained_config_archive_map = LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "bert"

    def __init__(self, max_2d_position_embeddings=1024, **kwargs):
        super().__init__(**kwargs)
        self.max_2d_position_embeddings = max_2d_position_embeddings


class LayoutlmModel(BertModel):

    config_class = LayoutlmConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(LayoutlmModel, self).__init__(config)
        self.embeddings = Embeddings2D(config)
        self.init_weights()

    def forward(self, input_ids, bbox, attention_mask=None, token_type_ids=None, position_ids=None,
                                            head_mask=None, inputs_embeds=None, encoder_hidden_states=None,
                                                                                encoder_attention_mask=None,):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask [batch_size, 1, 1, to_seq_length] from a 2D tensor mask.
        # So, we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention used in OpenAI GPT, 
        # we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for masked positions, 
        # this op will create a tensor which is 0.0 for positions to attend and -10_000.0 for masked positions.
        # Since we are adding it to raw scores before softmax, this is effectively the same as removing these entirely.
        extended_attention_mask =        extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1))  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to float if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, bbox, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)




