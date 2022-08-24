import os

from pathlib import Path
from copy import deepcopy

import transformers
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from src.models.layers import Embeddings2D, BertEncoder2D
from src.models.heads.dependency.SPADE.postprocess import gen_parses, pred_label
from src.models.heads.dependency.SPADE.periprocess import (
    embed_fields, select_header_vec, collect_features_batchwise, get_pretrained_transformer, 
    get_char_for_detokenization, get_local_rank, update_parts_of_model, 
    check_consistency_between_backbone_and_encoder, RelationTaggerUtils as RLUtils
)


class Encoder(pl.LightningModule):

    def __init__(self, hparam, path_data_folder):
        super().__init__()

        # Augment config
        transformer_cfg = self.get_transformer_config(hparam.encoder_config_name, 
                                                      hparam.encoder_backbone_tweak_tag,
                                                      path_data_folder,)
        transformer_cfg.pre_layer_norm = hparam.pre_layer_norm
        transformer_cfg.no_rel_attention = hparam.no_rel_attention
        transformer_cfg.trainable_rel_emb = hparam.trainable_rel_emb

        self.transformer_cfg = transformer_cfg

        self.embeddings = Embeddings2D(transformer_cfg, hparam.n_dist_unit,
                                                        hparam.n_char_unit,
                                                        hparam.input_embedding_components,)

        if hparam.encoder_backbone_name in ["bert-base-multilingual-cased"]:
            self.encoder = BertEncoder2D(transformer_cfg)
        else:
            raise NotImplementedError

        # rn embedding. rn stands for "relative, normalized"
        n_pos = hparam.n_dist_unit * 2 + 1
        self.n_dist_unit = hparam.n_dist_unit

        # check dimension compatibility
        assert transformer_cfg.hidden_size % 4 == 0
        quater_of_hidden_size = int(transformer_cfg.hidden_size / 4)
        if hparam.trainable_rel_emb:
            from models.layers import PositionalEncodingLearnable as PositionalEncoding
        else:
            from models.layers import PositionalEncoding

        self.rn_center_x_emb = PositionalEncoding(quater_of_hidden_size, max_seq_len=n_pos)
        self.rn_center_y_emb = PositionalEncoding(quater_of_hidden_size, max_seq_len=n_pos)
        self.rn_angle_emb    = PositionalEncoding(quater_of_hidden_size, max_seq_len=hparam.n_angle_unit)
        self.rn_dist_emb     = PositionalEncoding(quater_of_hidden_size, max_seq_len=hparam.n_dist_unit)

    def get_transformer_config(self, encoder_config_name, encoder_backbone_tweak_tag, path_data_folder):
        config_path = Path(path_data_folder) / "model/backbones" / encoder_config_name / encoder_backbone_tweak_tag / "config.json"
        transformer_cfg = transformers.BertConfig.from_json_file(config_path)
        return transformer_cfg

    def forward(self, text_tok_ids, rn_center_x_toks, rn_center_y_toks, rn_dist_toks, rn_angle_toks,
                    vertical_toks, char_size_toks, header_toks, token_type_ids=None, attention_mask=None,):
        if attention_mask is None:
            attention_mask = torch.ones_like(text_tok_ids)

        extended_attention_mask =                 attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask =        extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        input_vectors = self.embeddings(text_tok_ids,
                                 rn_center_x_toks,              # [B, len, len]
                                 rn_center_y_toks,
                                    vertical_toks,              # [B, len]
                                   char_size_toks,              # [B, len]
                                      header_toks,              # [B, len]
                            token_type_ids=token_type_ids,      # [B, len]
                                        )
        rn_emb = self.get_rn_emb(rn_center_x_toks,  # [B, len, len]
                                 rn_center_y_toks,
                                 rn_dist_toks,      # [B, len, len]
                                 rn_angle_toks,)

        all_encoder_layers = self.encoder(input_vectors, rn_emb, extended_attention_mask)
        # sequence_output = all_encoder_layers[-1]

        return all_encoder_layers

    def get_rn_emb(self, x, y, dist, angle):
        if self.transformer_cfg.trainable_rel_emb:
            return (self.rn_center_x_emb(x + self.n_dist_unit),
                    self.rn_center_y_emb(y + self.n_dist_unit),
                    self.rn_dist_emb(dist),
                    self.rn_angle_emb(angle),)
        else:
            return (self.rn_center_x_emb(x),
                    self.rn_center_y_emb(y),
                    self.rn_dist_emb(dist),
                    self.rn_angle_emb(angle),)


class Decoder(pl.LightningModule):

    def __init__(self, input_size, hidden_size, n_relation_type, fields, token_lv_boxing,
                        include_second_order_relations, vi_params=None,):
        super().__init__()

        # define metric for relation tagging
        self.n_relation_type = n_relation_type
        self.n_fields = len(fields)

        self.token_lv_boxing = token_lv_boxing
        self.h_pooler = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True) for _ in range(n_relation_type)])
        self.t_pooler = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True) for _ in range(n_relation_type)])
        self.W_label = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range((n_relation_type)*2)]) # include edge (+1). 0-score and 1-score (*2)

        # define embedding for each fields and [end] special tokens
        self.sph_emb = nn.ModuleList([nn.Embedding(self.n_fields, hidden_size) for _ in range(n_relation_type)])

        self.include_second_order_relations = include_second_order_relations
        if include_second_order_relations:
            self.n_vi_iter = vi_params["n_vi_iter"]
            self.do_gp = vi_params["do_gp"]
            self.do_sb = vi_params["do_sb"]

            self.ht_pooler = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True) for _ in range(n_relation_type)])
            self.W_uni = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range((n_relation_type)*1)])
            self.W_gp = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range((n_relation_type)*3)])    # grand parents            
            self.W_sb = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range((n_relation_type)*3)])    # sibling

        # special token header embedding
        self.initializer_range = 0.02  # std

    def forward(self, encoded, header_ids, lmax_boxes):

        batch_size, input_len, idim = encoded.shape
        l_boxes = header_ids.sum(dim=1)

        enc_header = select_header_vec(batch_size, lmax_boxes, header_ids, l_boxes, idim, encoded, self.token_lv_boxing,)

        lmax_units = input_len if self.token_lv_boxing else lmax_boxes

        # get score
        if self.include_second_order_relations:
            # initialize score
            unary_score = torch.zeros([batch_size,
                                       self.n_relation_type,
                                       2,  # Z = 0 or 1
                                       lmax_units + self.n_fields,
                                       lmax_units,])

            score = torch.zeros([batch_size,
                                 self.n_relation_type,  # +1 for edge
                                 2,  # Z = 0 or 1
                                 lmax_units + self.n_fields,
                                 lmax_units,])

            ternary_score_gp = torch.zeros([batch_size,
                                            self.n_relation_type,
                                            lmax_units + self.n_fields,
                                            lmax_units,
                                            lmax_units,])

            ternary_score_sb = torch.zeros([batch_size,
                                            self.n_relation_type,
                                            lmax_units + self.n_fields,
                                            lmax_units,
                                            lmax_units,])

            # calculate score.
            for i_label in range(self.n_relation_type):  # +1 for edge
                enc_header_h = self.h_pooler[i_label](enc_header)
                enc_header_t = self.t_pooler[i_label](enc_header)
                enc_header_ht = self.ht_pooler[i_label](enc_header)
                enc_sp = embed_fields(self.sph_emb[i_label], self.n_fields, batch_size)  # [bS, n_sp, dim]
                enc_header_h_all = torch.cat([enc_sp, enc_header_h], dim=1)

                unary_score[:, i_label, 1, :, :] = torch.matmul(enc_header_h_all,  # [batch, n_field + len_box, dim]
                                            self.W_uni[i_label](enc_header_t).transpose(1, 2),)

                # second order score
                if self.do_gp[i_label]:
                    # grand parents
                    # s_ij,jk : i-> j -? k

                    g0_gp = self._gen_g_vector(self.W_gp, 3 * i_label    , enc_header_h_all)
                    g1_gp = self._gen_g_vector(self.W_gp, 3 * i_label + 1, enc_header_ht)
                    g2_gp = self._gen_g_vector(self.W_gp, 3 * i_label + 2, enc_header_t)

                    ternary_score_gp[:, i_label, :, :, :] = torch.einsum("bid,bjd,bkd->bijk", g0_gp, g1_gp, g2_gp)

                if self.do_sb[i_label]:
                    # sibling
                    # s_ij,ik: i->j, i->k
                    g0_sb = self._gen_g_vector(self.W_sb, 3 * i_label, enc_header_h_all)
                    g1_sb = self._gen_g_vector(self.W_sb, 3 * i_label + 1, enc_header_t)
                    g2_sb = self._gen_g_vector(self.W_sb, 3 * i_label + 2, enc_header_t)

                    ternary_score_sb[:, i_label, :, :, :] = torch.einsum("bid,bjd,bkd->bijk", g0_sb, g1_sb, g2_sb)

            # VI now
            score[:] = unary_score[:]
            for i_vi in range(self.n_vi_iter):
                q_value = F.softmax(score, dim=2)
                
                # calculate F for Z=1 case
                F_value = self.get_F_value(q_value, ternary_score_sb, ternary_score_gp, self.n_fields, self.do_sb, self.do_gp,)

                # update Q
                # score[:, :, 0, :, :] = 0
                score = unary_score + F_value

            # reshape q_value for the consistency with zeroth-order
            score = score.view(batch_size, 2 * (self.n_relation_type), lmax_units + self.n_fields, lmax_units,)

        else:
            score = torch.zeros([batch_size,
                                 2 * (self.n_relation_type),  # +1 for edge
                                 lmax_units + self.n_fields,
                                 lmax_units,]).type_as(enc_header)

            for i_label in range(self.n_relation_type):  # +1 for edge
                enc_header_h = self.h_pooler[i_label](enc_header)
                enc_header_t = self.t_pooler[i_label](enc_header)
                enc_sp = embed_fields(self.sph_emb[i_label], self.n_fields, batch_size)  # [bS, n_sp, dim]

                score[:, 2 * i_label, :, :] = torch.matmul(
                    torch.cat([enc_sp, enc_header_h], dim=1),  # [batch, n_field + len_box, dim]
                    self.W_label[2 * i_label](enc_header_t).transpose(1, 2),
                )

                score[:, 2 * i_label + 1, :, :] = torch.matmul(
                    torch.cat([enc_sp, enc_header_h], dim=1),  # [batch, n_field + len_box, dim]
                    self.W_label[2 * i_label + 1](enc_header_t).transpose(1, 2),
                )

        return score

    @staticmethod
    def _gen_g_vector(W, i_type, vec):
        return W[i_type](vec)

    @staticmethod
    def get_F_value(q_value, ternary_score_sb, ternary_score_gp, n_fields, do_sb, do_gp):

        batch_size, n_edge_type, n_cases, n_row, n_col = q_value.shape
        assert n_cases == 2

        if sum(do_gp) == n_edge_type and sum(do_sb) == n_edge_type:
            F_value_sb = torch.einsum("bnik,bnijk->bnij", q_value[:, :, 1,         :, :], ternary_score_sb)
            F_value_gpA = torch.einsum("bnjk,bnijk->bnij", q_value[:, :, 1, n_fields:, :], ternary_score_gp)
            F_value_gpB = torch.zeros([batch_size, n_edge_type, n_row, n_col])
            F_value_gpB[:, :, n_fields:, :] = torch.einsum("bnki,bnkij->bnij", q_value[:, :, 1, :, :], ternary_score_gp)

            F_value = F_value_sb + F_value_gpA + F_value_gpB

        else:
            F_value = torch.zeros([batch_size, n_edge_type, n_row, n_col])
            for i_label in range(n_edge_type):
                if do_sb[i_label]:
                    F_value_sb1 = torch.einsum("bik,bijk->bij", q_value[:, i_label, 1, :, :],
                                                       ternary_score_sb[:, i_label, :, :, :],)
                    F_value[:, i_label, :, :] = F_value[:, i_label, :, :] + F_value_sb1

                if do_gp[i_label]:
                    F_value_gpA1 = torch.einsum("bjk,bijk->bij", q_value[:, i_label, 1, n_fields:, :],
                                                        ternary_score_gp[:, i_label, :,         :, :],)

                    F_value_gpB1 = torch.zeros([batch_size, n_row, n_col])
                    F_value_gpB1[:, n_fields:, :] = torch.einsum("bki,bkij->bij", q_value[:, i_label, 1, :, :],
                                                                         ternary_score_gp[:, i_label, :, :, :],)

                    F_value[:, i_label, :, :] = F_value[:, i_label, :, :] + F_value_gpA1
                    F_value[:, i_label, :, :] = F_value[:, i_label, :, :] + F_value_gpB1

        zero_value = torch.zeros_like(F_value)
        F_value = torch.cat([zero_value.unsqueeze(2), F_value.unsqueeze(2)], dim=2)

        return F_value


class RelationTagger(pl.LightningModule):

    def __init__(self, hparam, tparam, iparam, path_data_folder, verbose=False):
        super().__init__()

        self.name   = hparam.model_name
        self.hparam = hparam
        self.tparam = tparam
        self.iparam = iparam
        self.verbose = verbose

        self.task     = hparam.task
        self.task_lan = hparam.task_lan
        self.fields   = hparam.fields
        self.field_rs = hparam.field_representers
        self.n_fields = len(hparam.fields)
        self.max_input_len = hparam.max_input_len
        self.input_split_overlap_len = hparam.input_split_overlap_len
        self.encoder_layer_ids_used_in_decoder = hparam.encoder_layer_ids_used_in_decoder

        self.encoder_layer = self.gen_encoder_layer(hparam, path_data_folder)  # encoder
        self.decoder_layer = self.gen_decoder_layer(hparam, self.encoder_layer.transformer_cfg)

        self.parse_refine_options = {
            "refine_parse": self.iparam.refine_parse,
            "allow_small_edit_distance": self.iparam.allow_small_edit_distance,
            "task_lan": self.task_lan,
            "unwanted_fields": self.iparam.unwanted_fields,
        }

        self.char_for_detokenization = get_char_for_detokenization(hparam.encoder_backbone_name)

    def forward(self, text_tok_ids, rn_center_x_toks, rn_center_y_toks, rn_dist_toks, rn_angle_toks,
                    vertical_toks, char_size_toks, header_toks, n_seps, i_toks, j_toks, l_toks, lmax_toks, lmax_boxes,):
        
        # splits
        header_ori_toks = deepcopy(header_toks)
        encoded = self.encoder_forward(text_tok_ids, rn_center_x_toks, rn_center_y_toks, rn_dist_toks, rn_angle_toks,
                                    vertical_toks, char_size_toks, header_toks, n_seps, i_toks, j_toks, l_toks, lmax_toks,)

        # decoding
        score = self.decoder_layer(encoded, header_ori_toks, lmax_boxes)
        return score

    def encoder_forward(self, text_tok_ids, rn_center_x_toks, rn_center_y_toks, rn_dist_toks, rn_angle_toks,
                            vertical_toks, char_size_toks, header_toks, n_seps, i_toks, j_toks, l_toks, lmax_toks,):

        # 1. split features that have len > 512
        text_tok_ids, rn_center_x_toks, rn_center_y_toks, rn_dist_toks, rn_angle_toks, \
            vertical_toks, char_size_toks, header_toks = RLUtils.split_features(n_seps, i_toks, j_toks, self.max_input_len,
                                                                                text_tok_ids, rn_center_x_toks, rn_center_y_toks,
                                                                                              rn_dist_toks, rn_angle_toks,
                                                                                vertical_toks, char_size_toks, header_toks,)  

        # 2. encode each splitted feature
        encoded = []
        nmax_seps = max(n_seps)
        for i_sep in range(nmax_seps):
            attention_mask, l_mask = RLUtils.gen_input_mask(i_sep, l_toks, i_toks, j_toks, self.max_input_len)
            try:
                all_encoder_layer = self.encoder_layer(text_tok_ids[i_sep],
                                                   rn_center_x_toks[i_sep],
                                                   rn_center_y_toks[i_sep],
                                                       rn_dist_toks[i_sep],
                                                      rn_angle_toks[i_sep],
                                                      vertical_toks[i_sep],
                                                     char_size_toks[i_sep],
                                                        header_toks[i_sep], attention_mask=attention_mask,)

            except RuntimeError:
                print(f"i_sep = {i_sep + 1} / n_sep = {nmax_seps}")
                print("Fail to encode due to the memory limit.")
                print("The encoder output vectors set to zero.")
                if i_sep == 0:
                    print(f"Even single encoding faield!")
                    raise MemoryError
                else:
                    l_layer = self.encoder_layer.transformer_cfg.num_hidden_layers
                    all_encoder_layer = [torch.zeros_like(all_encoder_layer[0]) for _ in range(l_layer)]

            encoded1_part = RLUtils.get_encoded1_part(all_encoder_layer, self.max_input_len, self.input_split_overlap_len,
                                                      n_seps, i_sep, i_toks, j_toks, l_toks, self.encoder_layer_ids_used_in_decoder,)
            encoded.append(encoded1_part)

        # 3. Combine splited encoder outputs
        encoded = RLUtils.tensorize_encoded(encoded, l_toks, lmax_toks)
        return encoded

    # @gu.timeit
    def _run(self, mode, batch):

        # 1. Batchwise collection of features
        data_ids, image_urls, texts, text_toks, text_tok_ids, \
            labels, label_toks, rn_center_toks, rn_dist_toks, rn_angle_toks, \
            vertical_toks, char_size_toks, header_toks = collect_features_batchwise(batch)

        # 2. Calculate length for the padding.
        l_boxes = [len(x) for x in texts]
        l_tokens = [len(x) for x in text_toks]

        if self.hparam.token_lv_boxing:
            # Individual units are tokens.
            l_units = l_tokens
            label_units = label_toks
            text_units = text_toks
        else:
            # Individual units are text segments from OCR-detection-boxes.
            l_units = l_boxes
            label_units = labels
            text_units = texts

        # hot-fix
        # label_toks include junks when label is None.
        if labels[0] is None:
            label_units = labels

        lmax_boxes = max(l_boxes)

        # 3. Split data whose token length > 512
        n_seps, i_toks, j_toks, l_toks = RLUtils.get_split_param(text_toks, self.hparam.max_input_len,
                                                                            self.hparam.input_split_overlap_len,
                                                                type_informer_tensor=text_tok_ids[0],)

        # 4. get score
        batch_data_in = (text_tok_ids, rn_center_toks, rn_dist_toks, rn_angle_toks,
                        vertical_toks, char_size_toks, header_toks, n_seps, i_toks, j_toks, l_toks,)

        score = RLUtils.get_score(self, batch_data_in, lmax_boxes)

        # 5. prediction
        pr_label_units = pred_label(self.task, score, self.hparam.inferring_method, self.n_fields, l_units,)
        if labels[0] is not None:
            # 6. Generate gt parse
            parses, f_parses, text_unit_field_labels, f_parse_box_ids = gen_parses(self.task, self.fields, self.field_rs,
                                                                                    text_units, label_units, header_toks,
                                                                                    l_max_gen=self.hparam.l_max_gen_of_each_parse,
                                                                               max_info_depth=self.hparam.max_info_depth, strict=True,
                                                                              token_lv_boxing=self.hparam.token_lv_boxing,
                                                                                backbone_name=self.hparam.encoder_backbone_name,)
        else:
            parses = [None] * len(texts)
            f_parses = [None] * len(texts)
            text_unit_field_labels = [None] * len(texts)
            f_parse_box_ids = [None] * len(texts)

        # for the speed, set max serialization length small at initial stage.
        if mode == "train" and self.current_epoch <= 100:
            pr_l_max_gen = 2
        else:
            pr_l_max_gen = self.hparam.l_max_gen_of_each_parse

        # 7. Generate predicted parses
        pr_parses, pr_f_parses, pr_text_unit_field_labels, \
                   pr_f_parse_box_ids = gen_parses(self.task, self.fields, self.field_rs, text_units, pr_label_units, header_toks,
                                                    l_max_gen=pr_l_max_gen,
                                                    strict=False,
                                                     max_info_depth=self.hparam.max_info_depth,
                                                    token_lv_boxing=self.hparam.token_lv_boxing,
                                                      backbone_name=self.hparam.encoder_backbone_name,)

        results = {
            "data_ids": data_ids,
            "score": score,
            "text_units": text_units,
            "label_units": label_units,
            "pr_label_units": pr_label_units,
            "l_units": l_units,
            "parses": parses,
            "pr_parses": pr_parses,
            "text_unit_field_labels": text_unit_field_labels,
            "pr_text_unit_field_labels": pr_text_unit_field_labels,
            "f_parse_box_ids": f_parse_box_ids,
            "pr_f_parse_box_ids": pr_f_parse_box_ids,
        }

        return results

    def gen_encoder_layer(self, hparam, path_data_folder):
        # 1. Load pretrained transformer
        if hparam.encoder_backbone_is_pretrained:
            pretrained_transformer, \
            pretrained_transformer_cfg = get_pretrained_transformer(path_data_folder,
                                                                    hparam.encoder_backbone_name,
                                                                    hparam.encoder_backbone_tweak_tag,)

        # 2. Load model
        if hparam.encoder_type_name == "spade":
            # 2.1 Load encoder
            spatial_text_encoder = Encoder(hparam, path_data_folder)

            # 2.2 Initialize the subset of weights from the pretraiend transformer.
            if hparam.encoder_backbone_is_pretrained:
                print(f"pretrained {hparam.encoder_backbone_name} is used")
                check_consistency_between_backbone_and_encoder(pretrained_transformer_cfg, spatial_text_encoder.transformer_cfg)
                pretrained_transformer_state_dict = pretrained_transformer.state_dict()
                spatial_text_encoder = update_parts_of_model(parent_model_state_dict=pretrained_transformer_state_dict,
                                                                         child_model=spatial_text_encoder,
                                                                                rank=get_local_rank(),)
        else: 
            raise NotImplementedError

        return spatial_text_encoder


    def gen_decoder_layer(self, hparam, encoder_transformer_cfg):
        input_size = encoder_transformer_cfg.hidden_size * len(hparam.encoder_layer_ids_used_in_decoder)

        # 1. Load decoder
        if hparam.decoder_type == "spade":
            decoder_layer = Decoder(input_size,
                                    hparam.decoder_hidden_size,
                                    hparam.n_relation_type,
                                    hparam.fields,
                                    hparam.token_lv_boxing,
                                    hparam.include_second_order_relations,
                                    hparam.vi_params,)
        else:
            raise NotImplementedError
        return decoder_layer







