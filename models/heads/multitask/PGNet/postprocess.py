import sys
import numpy as np
import torch

from src.models.heads.multitask.PGNet.postprocess_common import get_dict
from src.models.heads.multitask.PGNet.postprocess_fast import generate_pivot_list_fast, restore_poly
from src.models.heads.multitask.PGNet.postprocess_slow import generate_pivot_list_slow, point_pair2poly, expand_poly_along_width


class PostProcess(object):

    output_names = ['center', 'border', 'direction', 'character']

    def __init__(self, character_dict_path, valid_set, score_thresh, mode: str = 'fast'):
        self.mode = mode
        self.valid_set = valid_set
        self.score_thresh = score_thresh
        self.Lexicon_Table = get_dict(character_dict_path)

        # c++ la-nms is faster, but only support python 3.5
        self.is_python35 = False
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            self.is_python35 = True

    def __call__(self, outs_dict, shape_list):
        p_dict = dict()
        for o in self.output_names:
            p_dict[o] = outs_dict[o][0]
            if isinstance(p_dict[o], torch.Tensor):
                p_dict[o] = p_dict[o].cpu().numpy()

        if self.mode == 'slow':
            polygon_list, keep_str_list = self.postprocess_slow(p_dict, shape_list)
        else:
            polygon_list, keep_str_list = self.postprocess_fast(p_dict, shape_list)
        return {
             'texts': keep_str_list,
            'points': polygon_list,
        }

    def postprocess_fast(self, p_dict, shape_list):
        src_h, src_w, ratio_h, ratio_w = shape_list[0]
        instance_yxs_list, seq_strs = generate_pivot_list_fast(p_dict['center'], p_dict['character'], p_dict['direction'],
                                                             self.Lexicon_Table, score_thresh=self.score_thresh)
        polygon_list, keep_str_list = restore_poly(instance_yxs_list, seq_strs, p_dict['border'], 
                                                    ratio_w, ratio_h, src_w, src_h, self.valid_set)
        return polygon_list, keep_str_list

    def postprocess_slow(self, p_dict, shape_list):
        src_h, src_w, ratio_h, ratio_w = shape_list[0]
        is_curved = self.valid_set == "totaltext"
        char_seq_idx_set, instance_yxs_list = generate_pivot_list_slow(p_dict['center'], p_dict['character'], p_dict['direction'],
                                                                    score_thresh=self.score_thresh, is_backbone=True, is_curved=is_curved)
        seq_strs = []
        for char_idx_set in char_seq_idx_set:
            pr_str = ''.join([self.Lexicon_Table[pos] for pos in char_idx_set])
            seq_strs.append(pr_str)
        polygon_list = []
        keep_str_list = []
        all_point_list = []
        all_point_pair_list = []
        for yx_center_line, keep_str in zip(instance_yxs_list, seq_strs):
            if len(yx_center_line) == 1:
                yx_center_line.append(yx_center_line[-1])

            offset_expand = 1.0
            if self.valid_set == 'totaltext':
                offset_expand = 1.2

            point_pair_list = []
            for b_id, y, x in yx_center_line:
                offset = p_dict['border'][:, y, x].reshape(2, 2)
                if offset_expand != 1.0:
                    offset_length = np.linalg.norm(offset, axis=1, keepdims=True)
                    expand_length = np.clip(offset_length * (offset_expand - 1), a_min=0.5, a_max=3.0)
                    offset_detal = offset / offset_length * expand_length
                    offset = offset + offset_detal
                ori_yx = np.array([y, x], dtype=np.float32)
                point_pair = (ori_yx + offset)[:, ::-1] * 4.0 / np.array([ratio_w, ratio_h]).reshape(-1, 2)
                point_pair_list.append(point_pair)

                all_point_list.append([int(round(x * 4.0 / ratio_w)), int(round(y * 4.0 / ratio_h))])
                all_point_pair_list.append(point_pair.round().astype(np.int32).tolist())

            detected_poly, _ = point_pair2poly(point_pair_list)
            detected_poly = expand_poly_along_width(detected_poly, shrink_ratio_of_width=0.2)
            detected_poly[:, 0] = np.clip(detected_poly[:, 0], a_min=0, a_max=src_w)
            detected_poly[:, 1] = np.clip(detected_poly[:, 1], a_min=0, a_max=src_h)

            if len(keep_str) < 2:
                continue

            keep_str_list.append(keep_str)
            detected_poly = np.round(detected_poly).astype('int32')
            if self.valid_set == 'partvgg':
                middle_point = len(detected_poly) // 2
                detected_poly = detected_poly[[0, middle_point-1, middle_point, -1], :]
                polygon_list.append(detected_poly)
            elif self.valid_set == 'totaltext':
                polygon_list.append(detected_poly)
            else:
                raise ValueError(f'valid_set = {self.valid_set} is not supported!')

        return polygon_list, keep_str_list

