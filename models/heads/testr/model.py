import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models.heads.testr.transformer import DeformableTransformer as Transformer
from models.layers.common import FeedForward
from models.layers.transformation.encodings import PositionalEncoding1D

from models.ops import sigmoid_offset, inverse_sigmoid_offset
from utils.batch import BatchedTensor, batchify_tensors


class TESTR(nn.Module):
    """
    Same as :class:`detectron2.modeling.ProposalNetwork`.
    Use one stage detector and a second stage for instance-wise prediction.
    """
    def __init__(self, cfg, backbone):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = backbone
        
        # fmt: off
        self.activation           = "relu"
        self.return_interm_decode = True
        self.num_classes          = 1
        self.d_model              = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.num_heads            = cfg.MODEL.TRANSFORMER.NUM_HEADS
        self.num_encoder_layers   = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers   = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dec_n_points         = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points         = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.dim_feedforward      = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout              = cfg.MODEL.TRANSFORMER.DROPOUT
        self.num_feature_levels   = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.num_proposals        = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.num_ctrl_points      = cfg.MODEL.TRANSFORMER.NUM_CTRL_POINTS
        self.pos_embed_scale      = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.max_text_len         = cfg.MODEL.TRANSFORMER.NUM_CHARS
        self.vocab_size           = cfg.MODEL.TRANSFORMER.VOCAB_SIZE
        self.sigmoid_offset   = not cfg.MODEL.TRANSFORMER.USE_POLYGON
        self.aux_loss             = cfg.MODEL.TRANSFORMER.AUX_LOSS

        self.text_pos_embed = PositionalEncoding1D(self.d_model, normalize=True, scale=self.pos_embed_scale)
        # fmt: on
        
        self.transformer = Transformer(
            dim_model=self.d_model, num_heads=self.num_heads, num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward,
            dropout=self.dropout, activation=self.activation, return_intermediate_dec=self.return_interm_decode,
            num_feature_levels=self.num_feature_levels, dec_n_points=self.dec_n_points, 
            enc_n_points=self.enc_n_points, num_proposals=self.num_proposals,
        )
        self.point_class = nn.Linear(in_features=self.d_model, out_features=self.num_classes)
        self.point_coord = FeedForward(input_dim=self.d_model, hidden_dims=[self.d_model] * 3, output_dim=2)
        self.bbox_coord = FeedForward(input_dim=self.d_model, hidden_dims=[self.d_model] * 3, output_dim=4)
        self.bbox_class = nn.Linear(in_features=self.d_model, out_features=self.num_classes)
        self.text_class = nn.Linear(in_features=self.d_model, out_features=self.vocab_size+1)

        # shared prior between instances (objects)
        self.point_embed = nn.Embedding(self.num_ctrl_points, self.d_model)
        self.text_embed = nn.Embedding(self.max_text_len, self.d_model)

                
        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            num_channels = [512, 1024, 2048]
            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),
                ))

            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, self.d_model),
                ))
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)

        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(num_channels[0], self.d_model, kernel_size=1),
                    nn.GroupNorm(32, self.d_model),)
            ])

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.bbox_class.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_coord.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.num_decoder_layers
        self.point_class = nn.ModuleList([self.point_class for _ in range(num_pred)])
        self.point_coord = nn.ModuleList([self.point_coord for _ in range(num_pred)])
        self.transformer.decoder.bbox_embed = None

        nn.init.constant_(self.bbox_coord.layers[-1].bias.data[2:], 0.0)
        self.transformer.bbox_class_embed = self.bbox_class
        self.transformer.bbox_coord_embed = self.bbox_coord

        self.to(self.device)


    def forward(self, samples: BatchedTensor):
        """ 
        The forward expects a BatchedTensor, which consists of:
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        
        It returns a dict with the following elements:
        - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes + 1)]
        - "pred_keypoints": The normalized keypoint coordinates for all queries, represented as
                            (x, y). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
                            See PostProcess for information on how to retrieve the unnormalized bounding box.
        - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = batchify_tensors(samples)
        features, pos = self.backbone(samples)

        if self.num_feature_levels == 1:
            features = [features[-1]]
            pos = [pos[-1]]

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = masks[0]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](BatchedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # n_points, embed_dim --> n_objects, n_points, embed_dim
        ctrl_point_embed = self.point_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)
        text_pos_embed = self.text_pos_embed(
                        self.text_embed.weight)[None, ...].repeat(self.num_proposals, 1, 1)
        text_embed = self.text_embed.weight[None, ...].repeat(self.num_proposals, 1, 1)

        hs, hs_text, init_reference, inter_references, enc_outputs_class, enc_outputs_coord = self.transformer(
            srcs, masks, pos, ctrl_point_embed, text_embed, text_pos_embed, text_mask=None)

        outputs_classes = []
        outputs_coords = []
        outputs_texts = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.point_class[lvl](hs[lvl])
            outputs_coord = self.point_coord[lvl](hs[lvl])
            if reference.shape[-1] == 2:
                outputs_coord += reference[:, :, None, :]
            elif reference.shape[-1] == 4:
                outputs_coord += reference[:, :, None, :2]
            else:
                raise ValueError(f'Last dim of reference must be 2 or 4, but get {reference.shape[-1]} instead.')
            outputs_coord = sigmoid_offset(outputs_coord, offset=self.sigmoid_offset)
            outputs_text = self.text_class(hs_text[lvl])

            outputs_texts.append(outputs_text)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_texts = torch.stack(outputs_texts)

        out = {
            'pred_logits' : outputs_classes[-1],
            'pred_points' : outputs_coords[-1],
            'pred_texts'  : outputs_texts[-1],
        }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_classes, outputs_coords, outputs_texts)

        enc_outputs_coord = enc_outputs_coord.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class, 
            'pred_boxes': enc_outputs_coord,
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_text):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_points': b, 'pred_texts': c}
                for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_text[:-1])]


