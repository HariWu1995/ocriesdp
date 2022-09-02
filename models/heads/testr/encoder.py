import torch
import torch.nn.functional as F
from torch import nn

from models.layers.attentions.deform import MultiScaleDeformableAttention as MSDeformAttn
from models.activations import get_activation_fn
from utils.models import duplicate


class DeformableTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                       n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self-attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_sa = nn.Dropout(dropout)
        self.laynorm_sa = nn.LayerNorm(d_model)

        # feed-forward
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.laynorm_ff = nn.LayerNorm(d_model)
        self.activation = get_activation_fn(activation)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, x):
        y = self.linear1(x)
        y = self.activation(y)
        y = self.dropout1(y)
        y = self.linear2(y)
        y = self.dropout2(y)

        # skip connection
        y = y + x
        y = self.laynorm_ff(y)
        return y

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self-attention
        src_ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = self.dropout_sa(src_) + src
        src = self.laynorm_sa(src)

        # feed-forward
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = duplicate(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points.append(ref)
        reference_points = torch.cat(reference_points, 1)
        reference_points =           reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

