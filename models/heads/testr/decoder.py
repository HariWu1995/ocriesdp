import torch
import torch.nn.functional as F
from torch import nn

from models.layers.attentions.deform import MultiScaleDeformableAttention as MSDeformAttn
from models.activations import get_activation_fn
from models.ops import inverse_sigmoid
from utils.models import duplicate


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                       n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_ca = nn.Dropout(dropout)
        self.laynorm_ca = nn.LayerNorm(d_model)

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_sa = nn.Dropout(dropout)
        self.laynorm_sa = nn.LayerNorm(d_model)

        # Feed-forward
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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # Self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt_ = self.self_attn(q.transpose(0, 1), 
                              k.transpose(0, 1), 
                            tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout_sa(tgt_)
        tgt = self.laynorm_sa(tgt)

        # Cross-attention
        tgt_ = self.with_pos_embed(tgt, query_pos)
        tgt_ = self.cross_attn(tgt_, reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout_ca(tgt_)
        tgt = self.laynorm_ca(tgt)

        # Feed-forward
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers: int = 1, return_intermediate: bool=False):
        super().__init__()
        self.layers = duplicate(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # hack implementation for iterative bounding box refinement and 2-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                      query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate_loc = []
        intermediate_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * \
                                            torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            elif reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            else:
                raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.')
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                elif reference_points.shape[-1] == 2:
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate_loc.append(output)
                intermediate_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate_loc), torch.stack(intermediate_points)

        return output, reference_points

class DeformableCompositeTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                        n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        ### Attention for location branch
        # cross-attention
        self.cross_attn_loc = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_ca_loc = nn.Dropout(dropout)
        self.laynorm_ca_loc = nn.LayerNorm(d_model)

        # self-attention (intra)
        self.intra_attn_loc = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_ia_loc = nn.Dropout(dropout)
        self.laynorm_ia_loc = nn.LayerNorm(d_model)

        # self-attention (inter)
        self.inter_attn_loc = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_ir_loc = nn.Dropout(dropout)
        self.laynorm_ir_loc = nn.LayerNorm(d_model)

        # feed-forward
        self.linear1_loc = nn.Linear(d_model, d_ffn)
        self.dropout1_loc = nn.Dropout(dropout)
        self.linear2_loc = nn.Linear(d_ffn, d_model)
        self.dropout2_loc = nn.Dropout(dropout)
        self.laynorm_ff_loc = nn.LayerNorm(d_model)
        self.activation_loc = get_activation_fn(activation)

        ### Attention (factorized) for text branch
        # cross-attention
        self.cross_attn_text = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout_ca_text = nn.Dropout(dropout)
        self.laynorm_ca_text = nn.LayerNorm(d_model)

        # attention between text embeddings belonging to the same object query
        self.intra_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_ia_text = nn.Dropout(dropout)
        self.laynorm_ia_text = nn.LayerNorm(d_model)

        # attention between text embeddings on the same spatial position of different objects
        self.inter_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_ir_text = nn.Dropout(dropout)
        self.laynorm_ir_text = nn.LayerNorm(d_model)

        # ffn
        self.linear1_text = nn.Linear(d_model, d_ffn)
        self.dropout1_text = nn.Dropout(dropout)
        self.linear2_text = nn.Linear(d_ffn, d_model)
        self.dropout2_text = nn.Dropout(dropout)
        self.laynorm_ff_text = nn.LayerNorm(d_model)
        self.activation_text = get_activation_fn(activation)

        # TODO: different embedding dim for text/loc?

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn_loc(self, x):
        y = self.linear1_loc(x)
        y = self.activation_loc(y)
        y = self.dropout1_loc(y)
        y = self.linear2_loc(y)
        y = self.dropout2_loc(y)

        # skip connection
        y = y + x
        y = self.laynorm_ff_loc(y)
        return y

    def forward_ffn_text(self, x):
        y = self.linear1_text(x)
        y = self.activation_text(y)
        y = self.dropout1_text(y)
        y = self.linear2_text(y)
        y = self.dropout2_text(y)

        # skip connection
        y = y + x
        y = self.laynorm_ff_text(y)
        return y

    def forward(self, tgt_loc, query_pos_loc, tgt_text, query_pos_text, reference_points, src, src_spatial_shapes, level_start_index, 
                      src_padding_mask=None, text_padding_mask=None):
        """
        Params:
        -------
        # tgt_loc:            batch_size, n_objects, n_points, embed_dim
        # query_pos_loc:      batch_size, n_objects, n_points, embed_dim
        # tgt_text:           batch_size, n_objects, n_words, embed_dim
        # query_pos_text:     batch_size, n_objects, n_words, embed_dim
        # text_padding_mask:  batch_size, n_objects, n_words
        """
        # location branch - self-attention (intra)
        q_intra = k_intra = self.with_pos_embed(tgt_loc, query_pos_loc)
        tgt_ = self.intra_attn_loc(q_intra.flatten(0, 1).transpose(0, 1), 
                                   k_intra.flatten(0, 1).transpose(0, 1), 
                                   tgt_loc.flatten(0, 1).transpose(0, 1),)[0].transpose(0, 1).reshape(q_intra.shape)
        tgt_loc = tgt_loc + self.dropout_ia_loc(tgt_)
        tgt_loc = self.laynorm_ia_loc(tgt_loc)

        # location branch - self-attention (inter)
        q_inter = k_inter = tgt_inter = torch.swapdims(tgt_loc, 1, 2)
        tgt_inter_ = self.inter_attn_loc(q_inter.flatten(0, 1).transpose(0, 1), 
                                         k_inter.flatten(0, 1).transpose(0, 1), 
                                       tgt_inter.flatten(0, 1).transpose(0, 1),)[0].transpose(0, 1).reshape(q_inter.shape)
        tgt_inter = tgt_inter + self.dropout_ir_loc(tgt_inter_)
        tgt_inter = self.laynorm_ia_loc(tgt_inter)
        tgt_inter = torch.swapdims(tgt_inter, 1, 2)

        # location branch - cross-attention
        reference_points_loc = reference_points[:, :, None, :, :].repeat(1, 1, tgt_inter.shape[2], 1, 1)
        tgt_cross = self.with_pos_embed(tgt_inter, query_pos_loc)
        tgt_ = self.cross_attn_loc(tgt_cross.flatten(1, 2), 
                        reference_points_loc.flatten(1, 2),
                                   src, src_spatial_shapes, level_start_index, src_padding_mask).reshape(tgt_inter.shape)
        tgt_inter = tgt_inter + self.dropout_ca_loc(tgt_)
        tgt_loc = self.laynorm_ca_loc(tgt_inter)

        # text branch - intra self-attention (word-wise)
        q_text = k_text = self.with_pos_embed(tgt_text, query_pos_text)
        tgt_text_ = self.intra_attn_text(q_text.flatten(0, 1).transpose(0, 1), 
                                         k_text.flatten(0, 1).transpose(0, 1),
                                       tgt_text.flatten(0, 1).transpose(0, 1),
                                           text_padding_mask.flatten(0, 1) 
                                        if text_padding_mask is not None else None)[0].transpose(0, 1).reshape(tgt_text.shape)
        tgt_text = self.dropout_ia_text(tgt_text_) + tgt_text
        tgt_text = self.laynorm_ia_text(tgt_text)

        # text branch - inter self-attention (object-wise)
        if text_padding_mask is not None:
            text_padding_mask = torch.swapdims(text_padding_mask, 1, 2).flatten(0, 1)
        q_inter = k_inter = tgt_inter = torch.swapdims(tgt_text, 1, 2)
        tgt_inter_ = self.inter_attn_text(q_inter.flatten(0, 1).transpose(0, 1),
                                          k_inter.flatten(0, 1).transpose(0, 1),
                                        tgt_inter.flatten(0, 1).transpose(0, 1),
                                        text_padding_mask,)[0].transpose(0, 1).reshape(q_inter.shape)
        tgt_inter = self.dropout_ir_text(tgt_inter_) + tgt_inter
        tgt_inter = self.laynorm_ir_text(tgt_inter)
        tgt_inter = torch.swapdims(tgt_inter, 1, 2)

        # text branch - cross-attention
        reference_points_text = reference_points[:, :, None, :, :].repeat(1, 1, tgt_inter.shape[2], 1, 1)
        tgt_cross = self.with_pos_embed(tgt_inter, query_pos_text)
        tgt_text_ = self.cross_attn_text(tgt_cross.flatten(1, 2),
                             reference_points_text.flatten(1, 2),
                                src, src_spatial_shapes, level_start_index, src_padding_mask).reshape(tgt_inter.shape)
        
        tgt_text = self.dropout_ca_text(tgt_text_) + tgt_inter
        tgt_text = self.laynorm_ca_text(tgt_text)

        # ffn
        tgt_loc = self.forward_ffn_loc(tgt_loc)
        tgt_text = self.forward_ffn_text(tgt_text)

        return tgt_loc, tgt_text


class DeformableCompositeTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate: bool=False):
        super().__init__()
        self.layers = duplicate(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

        # hack implementation for iterative bounding box refinement and 2-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt_loc, tgt_text, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                      query_pos_loc=None, query_pos_text=None, src_padding_mask=None, text_padding_mask=None):
        
        output_loc, output_text = tgt_loc, tgt_text

        intermediate_loc = []
        intermediate_text = []
        intermediate_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            elif reference_points.shape[-1] == 2:
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            else:
                raise ValueError(f'Last dim of reference_points must be 2 or 4, but get {reference_points.shape[-1]} instead.')
            output_loc, output_text = layer(output_loc, query_pos_loc, output_text, query_pos_text, reference_points_input, 
                                            src, src_spatial_shapes, src_level_start_index, src_padding_mask, text_padding_mask)

            if self.return_intermediate:
                intermediate_loc.append(output_loc)
                intermediate_text.append(output_text)
                intermediate_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate_loc), \
                   torch.stack(intermediate_text), \
                   torch.stack(intermediate_points)

        return output_loc, output_text, reference_points


