# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.container import T

from models.layers.attentions.deform import MultiScaleDeformableAttention as MSDeformAttn


PI = math.pi


class DeformableTransformer(nn.Module):

    def __init__(self, dim_model=256, num_heads=8, dim_ff=1024, dropout=0.1, activation="relu", 
                 num_encoder_layers=6, enc_n_points=4, 
                 num_decoder_layers=6, dec_n_points=4, 
                 num_feature_levels=4, num_proposals=300, 
                 return_interm_decode: bool=False, use_composite_decoder: bool = True):
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.num_proposals = num_proposals

        from models.heads.testr.encoder import (
            DeformableTransformerEncoder as Encoder, 
            DeformableTransformerEncoderLayer as EncoderLayer
        )

        if use_composite_decoder:
            from models.heads.testr.decoder import (
                DeformableCompositeTransformerDecoder as Decoder, 
                DeformableCompositeTransformerDecoderLayer as DecoderLayer
            )
        else:
            from models.heads.testr.decoder import (
                DeformableTransformerDecoder as Decoder, 
                DeformableTransformerDecoderLayer as DecoderLayer
            )

        encoder_layer = EncoderLayer(dim_model, dim_ff, dropout, activation, num_feature_levels, num_heads, enc_n_points)
        decoder_layer = DecoderLayer(dim_model, dim_ff, dropout, activation, num_feature_levels, num_heads, dec_n_points)

        self.encoder = Encoder(encoder_layer, num_encoder_layers)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, return_interm_decode)
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, dim_model))
        
        self.bbox_class_embed = None
        self.bbox_coord_embed = None
        self.enc_output = nn.Linear(dim_model, dim_model)
        self.enc_output_norm = nn.LayerNorm(dim_model)
        self.pos_trans = nn.Linear(dim_model, dim_model)
        self.pos_trans_norm = nn.LayerNorm(dim_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 64
        temperature = 10_000
        scale = 2 * PI

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / num_pos_feats)

        # N, L, 4
        proposals = proposals.sigmoid() * scale
        
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), 
                           pos[:, :, :, 1::2].cos(),), dim=4).flatten(2)
        return pos

    def generate_encoder_output_proposals(self, memory, padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), 
                              grid_y.unsqueeze(-1),], dim=-1)

            scale = torch.cat([valid_W.unsqueeze(-1), 
                               valid_H.unsqueeze(-1),], dim=1).view(N_, 1, 1, 2)

            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        proposals = torch.cat(proposals, 1)
        proposals_valid = ((proposals > 0.01) & (proposals < 0.99)).all(-1, keepdim=True)
        proposals = torch.log(proposals / (1 - proposals))
        proposals = proposals.masked_fill(padding_mask.unsqueeze(-1), float('inf'))
        proposals = proposals.masked_fill(~proposals_valid, float('inf'))

        memory = memory.masked_fill(padding_mask.unsqueeze(-1), float(0))
        memory = memory.masked_fill(~proposals_valid, float(0))
        memory = self.enc_output_norm(self.enc_output(memory))
        return memory, proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], dim=-1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed, text_embed, text_pos_embed, text_mask=None):
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            pos_embed_flatten.append(pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        pos_embed_flatten = torch.cat(pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), 
                                       spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten, spatial_shapes, level_start_index, valid_ratios, pos_embed_flatten, mask_flatten
        )

        # prepare input for decoder
        bs, _, c = memory.shape
        output_memory, output_proposals = self.generate_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

        # hack implementation for 2-stage Deformable DETR
        enc_outputs_class = self.bbox_class_embed(output_memory)
        enc_outputs_coord = self.bbox_coord_embed(output_memory) + output_proposals

        topk = self.num_proposals
        topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
        topk_coords = torch.gather(enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords.detach()
        reference_points = topk_coords.sigmoid()
        init_reference_out = reference_points
        query_pos = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords)))
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1, -1)
        query_pos = query_pos[:, :, None, :].repeat(1, 1, query_embed.shape[2], 1)
        text_embed = text_embed.unsqueeze(0).expand(bs, -1, -1, -1)

        # decoder
        hs, hs_text, inter_references = self.decoder(
            query_embed, text_embed, reference_points, memory, spatial_shapes, 
            level_start_index, valid_ratios, query_pos, text_pos_embed, mask_flatten, text_mask
        )

        inter_references_out = inter_references
        return hs, hs_text, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord







