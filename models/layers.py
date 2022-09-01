from typing import *
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers.activations import ACT2FN


class Debug(nn.Module):
    
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f'{self.name}')
        print(f'type: {type(x)}, len: {len(x)}')
        print(f'shapes: {self._get_shape_recurse(x)}')
        return x

    def _get_shape_recurse(self, x):
        if isinstance(x, torch.Tensor):
            return x.shape
        return [self._get_shape_recurse(a) for a in x]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout: float=0.1, max_seq_len: int=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10_000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
 

class PositionalEncodingLearnable(nn.Module):

    def __init__(self, d_model, dropout: float=0.1, max_seq_len: int=100):
        super(PositionalEncodingLearnable, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layernorm = LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(0)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(-1).expand(x.size()[:2])
        x = x + self.pos_embed(pos)
        return self.dropout(self.layernorm(x))


class LayerNorm(nn.Module):
    """
    A layernorm module in the TF-style (epsilon inside the square root).
    """
    def __init__(self, d_model: int, variance: float=1e-11):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))
        self.variance = variance

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance)
        return self.gamma * x + self.beta


class MLP(nn.Module):

    activations = {
              'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(),
    }

    def __init__(self, in_dim: int, out_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None, layer_norm: bool = False,
                 dropout: Optional[float] = 0.0, activation: Optional[str] = 'relu'):
        super().__init__()
        layers = []

        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(self.activations.get(activation, nn.Identity()))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout:
                    layers.append(nn.Dropout(dropout))
                in_dim = dim

        if not out_dim:
            self.out_dim = hidden_dims[-1]
            layers.append(nn.Identity())
        else:
            self.out_dim = out_dim
            layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(x, 1))


class BidirLSTM(nn.Module):

    def __init__(self, lstm_config, mlp_config, padding_value: int = 0):
        super().__init__()
        self.padding_value = padding_value  # keys_vocab_cls.stoi['<pad>']
        self.lstm = nn.LSTM(**lstm_config)
        self.mlp = MLP(**mlp_config)

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
        logits = self.mlp(output) # (B, N*T, out_dim)
        return logits


class Conv2dDynamicSamePadding(nn.Conv2d):
    """
    2D Convolutions like TensorFlow, for a dynamic image size.
    The padding is operated in forward function by calculating dynamically.

    Tips for 'SAME' mode padding.
        Given the following:
            i: width or height
            s: stride
            k: kernel size
            d: dilation
            p: padding

        Output after Conv2d:
            o = floor( (i+p-((k-1)*d+1)) / s + 1 )

    If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    => p = (i-1)*s+((k-1)*d+1)-i
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride !!!
        pad_h = max((oh-1) * self.stride[0] + (kh-1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow-1) * self.stride[1] + (kw-1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """
    2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.

    With the same calculation as Conv2dDynamicSamePadding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh-1) * self.stride[0] + (kh-1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow-1) * self.stride[1] + (kw-1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class MaxPool2dDynamicSamePadding(nn.MaxPool2d):
    """
    2D MaxPooling like TensorFlow's 'SAME' mode, with a dynamic image size.
    The padding is operated in forward function by calculating dynamically.
    """
    def __init__(self, kernel_size, stride, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh-1) * self.stride[0] + (kh-1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow-1) * self.stride[1] + (kw-1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, 
                          pad_h // 2, pad_h - pad_h // 2])
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                            self.dilation, self.ceil_mode, self.return_indices)


class MaxPool2dStaticSamePadding(nn.MaxPool2d):
    """
    2D MaxPooling like TensorFlow's 'SAME' mode, with the given input image size.
    The padding mudule is calculated in construction function, then used in forward.
    """
    def __init__(self, kernel_size, stride, image_size=None, **kwargs):
        super().__init__(kernel_size, stride, **kwargs)
        self.stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride
        self.dilation = [self.dilation] * 2 if isinstance(self.dilation, int) else self.dilation
        self.kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh-1) * self.stride[0] + (kh-1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow-1) * self.stride[1] + (kw-1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, 
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.max_pool2d(x, self.kernel_size, self.stride, self.padding,
                         self.dilation, self.ceil_mode, self.return_indices)
        return x


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


class BertLayer2D(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention2D(config)
        self.intermediate = BertIntermediate2D(config)
        self.output = BertOutput2D(config)

    def forward(self, hidden_states, rn_emb, attention_mask):
        attention_output = self.attention(hidden_states, rn_emb, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertAttention2D(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.attention = BertSelfAttention2D(config)
        self.output = BertSelfOutput2D(config)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, input_tensor, rn_emb, attention_mask):
        if self.pre_layer_norm:
            input_tensor = self.LayerNorm(input_tensor)
        output_tensor = self.attention(input_tensor, rn_emb, attention_mask)
        attention_output = self.output(output_tensor, input_tensor)
        return attention_output


class BertSelfAttention2D(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.no_rel_attention = config.no_rel_attention
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads =                          config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sqrt_of_attention_hidden_size = math.sqrt(self.attention_head_size)
        if not self.no_rel_attention:
            # concat
            assert self.all_head_size == config.hidden_size
            assert self.all_head_size % 4 == 0
            quarter_of_hidden_size = int(self.all_head_size / 4)

            self.rn_dist     = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)
            self.rn_angle    = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)
            self.rn_center_x = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)
            self.rn_center_y = nn.Linear(quarter_of_hidden_size, quarter_of_hidden_size)

            self.query_rel = nn.Linear(config.hidden_size, self.all_head_size)
            self.key_bias = nn.Parameter(torch.zeros(config.num_attention_heads, self.attention_head_size,))
            self.rel_bias = nn.Parameter(torch.zeros(config.num_attention_heads, self.attention_head_size,))

        # self.initializer_range = 0.02
        # self.key_bias = nn.Parameter(torch.randn(config.num_attention_heads, self.attention_head_size) * self.initializer_range)
        # self.rel_bias = nn.Parameter(torch.randn(config.num_attention_heads, self.attention_head_size) * self.initializer_range)

    def transpose_for_scores(self, x):
        # [B, seq_len, hidden_size] -> [B, seq_len, num_attention_heads, attention_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size,)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [B, num_attention_heads, seq_len, attention_head_size]

    def transpose_for_scores_rn(self, x):
        # [B, seq_len, hidden_size] -> [B, seq_len, num_attention_heads, attention_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size,)
        x = x.view(*new_x_shape)
        return x.permute(0, 3, 1, 2, 4)

    def forward(self, hidden_states, rn_emb, attention_mask):
        # Embeddings
        rn_center_x_emb, rn_center_y_emb, rn_dist_emb, rn_angle_emb = rn_emb

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer   = self.key(  hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if not self.no_rel_attention:
            rn_center_x_emb = self.rn_center_x(rn_center_x_emb)
            rn_center_y_emb = self.rn_center_y(rn_center_y_emb)
            rn_dist_emb = self.rn_dist(rn_dist_emb)
            rn_angle_emb = self.rn_angle(rn_angle_emb)

            rn_emb_all = torch.cat([rn_center_x_emb, rn_center_y_emb, rn_dist_emb, rn_angle_emb], dim=-1)

            # mixed_rn_all_layer = self.rn_all(rn_emb_all)
            mixed_rn_all_layer = rn_emb_all
            rn_all_layer = self.transpose_for_scores_rn(mixed_rn_all_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.einsum("bhid,bhjd->bhij", query_layer, key_layer)

        # rn_dist
        if not self.no_rel_attention:
            mixed_query_rel_layer = self.query_rel(hidden_states)
            query_rel_layer = self.transpose_for_scores(mixed_query_rel_layer)

            attention_q_rn_all = torch.einsum("bhid,bhijd->bhij", query_rel_layer, rn_all_layer)
            attention_scores += attention_q_rn_all

            key_bias = self.key_bias.unsqueeze(0).unsqueeze(2).expand_as(key_layer)
            rel_bias = self.rel_bias.unsqueeze(0).unsqueeze(2).expand_as(key_layer)

            attention_key_bias = torch.einsum("bhid,bhjd->bhij", key_bias, key_layer)
            attention_scores += attention_key_bias

            attention_rel_bias = torch.einsum("bhid,bhijd->bhij", rel_bias, rn_all_layer)
            attention_scores += attention_rel_bias

        attention_scores = attention_scores / self.sqrt_of_attention_hidden_size
        attention_scores = (attention_scores + attention_mask)  # sort of multiplication in soft-max step. It is ~ -10_000

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might seem a bit unusual
        attention_probs = self.dropout(attention_probs)

        #    [B, num_attention_heads, seq_len, seq_len] * 
        #    [B, num_attention_heads, seq_len, attention_head_size]
        # -> [B, num_attention_heads, seq_len, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # -> [B, seq_len, num_attention_heads, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # [B, seq_len] + [all_head_size=hidden_size] -> [B, seq_len, all_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput2D(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.dense     = nn.Linear(   config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(  config.hidden_dropout_prob)

        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor
        if not self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertIntermediate2D(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, hidden_states):
        if self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput2D(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.dense     = nn.Linear(   config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(  config.hidden_dropout_prob)

        self.pre_layer_norm = config.pre_layer_norm

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor
        if not self.pre_layer_norm:
            hidden_states = self.LayerNorm(hidden_states)
        return hidden_states






