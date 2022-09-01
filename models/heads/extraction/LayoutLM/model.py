import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from models.backbones.layoutlm_v3 import (
    BertPreTrainedModel, LayoutlmConfig, LayoutlmModel,
    LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP, 
)


class TokenClassification(BertPreTrainedModel):

    config_class = LayoutlmConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = LayoutlmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, bbox, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        outputs = self.bert(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fn = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


class SequenceClassification(BertPreTrainedModel):

    config_class = LayoutlmConfig
    pretrained_model_archive_map = LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = LayoutlmModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, bbox, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):

        outputs = self.bert(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask,
                         token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1: 
                # regression
                loss_fn = MSELoss()
                loss = loss_fn(logits.view(-1), labels.view(-1))
            else:
                # classification
                loss_fn = CrossEntropyLoss()
                loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)




