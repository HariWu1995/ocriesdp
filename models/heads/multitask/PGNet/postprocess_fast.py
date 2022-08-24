from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import math

import numpy as np
from itertools import groupby
from skimage.morphology._skeletonize import thin

from src.models.heads.multitask.PGNet.postprocess_common import sort_and_expand_with_direction_v2


def instance_ctc_greedy_decoder(gather_info, logits_map, pts_num=4):
    """
    gather_info: [[x, y], [x, y] ...]
    logits_map: H x W X (n_chars + 1)
    """
    _, _, C = logits_map.shape
    ys, xs = zip(*gather_info)
    logits_seq = logits_map[list(ys), list(xs)]
    probs_seq = logits_seq
    labels = np.argmax(probs_seq, axis=1)
    dst_str = [k for k, v_ in groupby(labels) if k != C - 1]
    detal = len(gather_info) // (pts_num - 1)
    keep_idx_list = [0] + [detal * (i + 1) for i in range(pts_num - 2)] + [-1]
    keep_gather_list = [gather_info[idx] for idx in keep_idx_list]
    return dst_str, keep_gather_list


def ctc_decoder_for_image(gather_info_list, logits_map, Lexicon_Table, pts_num=6):
    """
    CTC decoder using multiple processes.
    """
    decoder_str = []
    decoder_xys = []
    for gather_info in gather_info_list:
        if len(gather_info) < pts_num:
            continue
        dst_str, xys_list = instance_ctc_greedy_decoder(
            gather_info, logits_map, pts_num=pts_num)
        dst_str_readable = ''.join([Lexicon_Table[idx] for idx in dst_str])
        if len(dst_str_readable) < 2:
            continue
        decoder_str.append(dst_str_readable)
        decoder_xys.append(xys_list)
    return decoder_str, decoder_xys


def point_pair2poly(point_pair_list):
    """
    Transfer vertical point_pairs into poly point in clockwise.
    """
    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2)


def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    ratio_pair = np.array(
        [[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
    p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
    p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
    return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])


def expand_poly_along_width(poly, shrink_ratio_of_width=0.3):
    """
    expand poly along width.
    """
    point_num = poly.shape[0]
    left_quad = np.array(
        [poly[0], poly[1], poly[-2], poly[-1]], dtype=np.float32)
    left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / \
                 (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    right_quad = np.array(
        [
            poly[point_num // 2 - 2], poly[point_num // 2 - 1],
            poly[point_num // 2], poly[point_num // 2 + 1]
        ],
        dtype=np.float32)
    right_ratio = 1.0 + shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / \
                  (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    poly[0] = left_quad_expand[0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2] = right_quad_expand[2]
    return poly


def restore_poly(instance_yxs_list, seq_strs, p_border, ratio_w, ratio_h, src_w,
                 src_h, valid_set):
    polygon_list = []
    keep_str_list = []
    for yx_center_line, keep_str in zip(instance_yxs_list, seq_strs):
        if len(keep_str) < 2:
            print('--> too short, {}'.format(keep_str))
            continue

        offset_expand = 1.0
        if valid_set == 'totaltext':
            offset_expand = 1.2

        point_pair_list = []
        for y, x in yx_center_line:
            offset = p_border[:, y, x].reshape(2, 2) * offset_expand
            ori_yx = np.array([y, x], dtype=np.float32)
            point_pair = (ori_yx + offset)[:, ::-1] * 4.0 / np.array(
                [ratio_w, ratio_h]).reshape(-1, 2)
            point_pair_list.append(point_pair)

        detected_poly = point_pair2poly(point_pair_list)
        detected_poly = expand_poly_along_width(
            detected_poly, shrink_ratio_of_width=0.2)
        detected_poly[:, 0] = np.clip(detected_poly[:, 0], a_min=0, a_max=src_w)
        detected_poly[:, 1] = np.clip(detected_poly[:, 1], a_min=0, a_max=src_h)

        keep_str_list.append(keep_str)
        if valid_set == 'partvgg':
            middle_point = len(detected_poly) // 2
            detected_poly = detected_poly[
                [0, middle_point - 1, middle_point, -1], :]
            polygon_list.append(detected_poly)
        elif valid_set == 'totaltext':
            polygon_list.append(detected_poly)
        else:
            raise ValueError(f'valid_set = {valid_set} is not supported!')
            
    return polygon_list, keep_str_list


def generate_pivot_list_fast(p_score,
                             p_char_maps,
                             f_direction,
                             Lexicon_Table,
                             score_thresh=0.5):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direction = f_direction.transpose(1, 2, 0)
    p_tcl_map = (p_score > score_thresh) * 1.0
    skeleton_map = thin(p_tcl_map.astype(np.uint8))
    instance_count, instance_label_map = cv2.connectedComponents(
        skeleton_map.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            if len(pos_list) < 3:
                continue

            pos_list_sorted = sort_and_expand_with_direction_v2(pos_list, f_direction, p_tcl_map)
            all_pos_yxs.append(pos_list_sorted)

    p_char_maps = p_char_maps.transpose([1, 2, 0])
    decoded_str, keep_yxs_list = ctc_decoder_for_image(
        all_pos_yxs, logits_map=p_char_maps, Lexicon_Table=Lexicon_Table)
    return keep_yxs_list, decoded_str






