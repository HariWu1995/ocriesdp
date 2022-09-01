from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import math

import numpy as np
from skimage.morphology._skeletonize import thin

from models.heads.multitask.PGNet.postprocess_common import (
    softmax, ctc_greedy_decoder, add_id, extract_main_direction, 
    sort_and_expand_with_direction_v2, sort_with_direction,
)


def point_pair2poly(point_pair_list):
    """
    Transfer vertical point_pairs into poly point in clockwise.
    """
    pair_length_list = []
    for point_pair in point_pair_list:
        pair_length = np.linalg.norm(point_pair[0] - point_pair[1])
        pair_length_list.append(pair_length)
    pair_length_list = np.array(pair_length_list)
    pair_info = (pair_length_list.max(), pair_length_list.min(),
                 pair_length_list.mean())

    point_num = len(point_pair_list) * 2
    point_list = [0] * point_num
    for idx, point_pair in enumerate(point_pair_list):
        point_list[idx] = point_pair[0]
        point_list[point_num - 1 - idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2), pair_info


def shrink_quad_along_width(quad, begin_width_ratio=0., end_width_ratio=1.):
    """
    Generate shrink_quad_along_width.
    """
    ratio_pair = np.array([[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
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
    left_ratio = -shrink_ratio_of_width * np.linalg.norm(left_quad[0] - left_quad[3]) / (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    right_quad = np.array([poly[point_num // 2 - 2], poly[point_num // 2 - 1],
                           poly[point_num // 2    ], poly[point_num // 2 + 1],], dtype=np.float32)
    right_ratio = 1.0 + \
        shrink_ratio_of_width * np.linalg.norm(right_quad[0] - right_quad[3]) / (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    poly[ 0] = left_quad_expand[ 0]
    poly[-1] = left_quad_expand[-1]
    poly[point_num // 2 - 1] = right_quad_expand[1]
    poly[point_num // 2    ] = right_quad_expand[2]
    return poly


def instance_ctc_greedy_decoder(gather_info, logits_map, keep_blank_in_idxs=True):
    """
    gather_info: [[x, y], [x, y] ...]
    logits_map: H x W X (n_chars + 1)
    """
    _, _, C = logits_map.shape
    ys, xs = zip(*gather_info)
    logits_seq = logits_map[list(ys), list(xs)]  # n x 96
    probs_seq = softmax(logits_seq)
    dst_str, keep_idx_list = ctc_greedy_decoder(probs_seq, blank=C-1, keep_blank_in_idxs=keep_blank_in_idxs)
    keep_gather_list = [gather_info[idx] for idx in keep_idx_list]
    return dst_str, keep_gather_list


def ctc_decoder_for_image(gather_info_list, logits_map, keep_blank_in_idxs=True):
    """
    CTC decoder using multiple processes.
    """
    decoder_results = []
    for gather_info in gather_info_list:
        res = instance_ctc_greedy_decoder(gather_info, logits_map, keep_blank_in_idxs=keep_blank_in_idxs)
        decoder_results.append(res)
    return decoder_results


def generate_pivot_list_curved(p_score, p_char_maps, f_direct, score_thresh=0.5, is_expand=True, is_backbone=False, image_id=0):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direct = f_direct.transpose(1, 2, 0)
    p_tcl_map = (p_score > score_thresh) * 1.0
    skeleton_map = thin(p_tcl_map)
    instance_count, \
    instance_label_map = cv2.connectedComponents(skeleton_map.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    center_pos_yxs = []
    end_points_yxs = []
    instance_center_pos_yxs = []
    pred_strs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            ### FIX-ME, eliminate outlier
            if len(pos_list) < 3:
                continue

            if is_expand:
                pos_list_sorted = sort_and_expand_with_direction_v2(pos_list, f_direct, p_tcl_map)
            else:
                pos_list_sorted, _ = sort_with_direction(pos_list, f_direct)
            all_pos_yxs.append(pos_list_sorted)

    # use decoder to filter backgroud points.
    p_char_maps = p_char_maps.transpose([1, 2, 0])
    decode_res = ctc_decoder_for_image(all_pos_yxs, logits_map=p_char_maps, keep_blank_in_idxs=True)
    for decoded_str, keep_yxs_list in decode_res:
        if is_backbone:
            keep_yxs_list_with_id = add_id(keep_yxs_list, image_id=image_id)
            instance_center_pos_yxs.append(keep_yxs_list_with_id)
            pred_strs.append(decoded_str)
        else:
            end_points_yxs.extend((keep_yxs_list[0], keep_yxs_list[-1]))
            center_pos_yxs.extend(keep_yxs_list)

    if is_backbone:
        return pred_strs, instance_center_pos_yxs
    else:
        return center_pos_yxs, end_points_yxs


def generate_pivot_list_horizontal(p_score, p_char_maps, f_direction, score_thresh=0.5, is_backbone=False, image_id=0):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direction = f_direction.transpose(1, 2, 0)
    p_tcl_map_bi = (p_score > score_thresh) * 1.0
    instance_count, \
    instance_label_map = cv2.connectedComponents(p_tcl_map_bi.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    center_pos_yxs = []
    end_points_yxs = []
    instance_center_pos_yxs = []

    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            ### FIX-ME, eliminate outlier
            if len(pos_list) < 5:
                continue

            # add rule here
            main_direction = extract_main_direction(pos_list, f_direction)  # y x
            reference_directin = np.array([0, 1]).reshape([-1, 2])  # y x
            is_h_angle = abs(np.sum(main_direction * reference_directin)) < math.cos(math.pi / 180 * 70)

            point_yxs = np.array(pos_list)
            max_y, max_x = np.max(point_yxs, axis=0)
            min_y, min_x = np.min(point_yxs, axis=0)
            is_h_len = (max_y - min_y) < 1.5 * (max_x - min_x)

            pos_list_final = []
            if is_h_len:
                xs = np.unique(xs)
                for x in xs:
                    ys = instance_label_map[:, x].copy().reshape((-1, ))
                    y = int(np.where(ys == instance_id)[0].mean())
                    pos_list_final.append((y, x))
            else:
                ys = np.unique(ys)
                for y in ys:
                    xs = instance_label_map[y, :].copy().reshape((-1, ))
                    x = int(np.where(xs == instance_id)[0].mean())
                    pos_list_final.append((y, x))

            pos_list_sorted, _ = sort_with_direction(pos_list_final, f_direction)
            all_pos_yxs.append(pos_list_sorted)

    # use decoder to filter backgroud points.
    p_char_maps = p_char_maps.transpose([1, 2, 0])
    decode_res = ctc_decoder_for_image(
        all_pos_yxs, logits_map=p_char_maps, keep_blank_in_idxs=True)
    for decoded_str, keep_yxs_list in decode_res:
        if is_backbone:
            keep_yxs_list_with_id = add_id(keep_yxs_list, image_id=image_id)
            instance_center_pos_yxs.append(keep_yxs_list_with_id)
        else:
            end_points_yxs.extend((keep_yxs_list[0], keep_yxs_list[-1]))
            center_pos_yxs.extend(keep_yxs_list)

    if is_backbone:
        return instance_center_pos_yxs
    else:
        return center_pos_yxs, end_points_yxs


def generate_pivot_list_slow(p_score, p_char_maps, f_direction, score_thresh=0.5, is_backbone=False, is_curved=True, image_id=0):
    """
    Warp all the function together.
    """
    if is_curved:
        return generate_pivot_list_curved(p_score, p_char_maps, f_direction, score_thresh=score_thresh, is_expand=True, is_backbone=is_backbone, image_id=image_id)
    else:
        return generate_pivot_list_horizontal(p_score, p_char_maps, f_direction, score_thresh=score_thresh, is_backbone=is_backbone, image_id=image_id)


def generate_pivot_list_tt_inference(p_score, p_char_maps, f_direction, score_thresh=0.5, is_backbone=False, is_curved=True, image_id=0):
    """
    return center point and end point of TCL instance; filter with the char maps;
    """
    p_score = p_score[0]
    f_direction = f_direction.transpose(1, 2, 0)
    p_tcl_map = (p_score > score_thresh) * 1.0
    skeleton_map = thin(p_tcl_map)
    instance_count, instance_label_map = cv2.connectedComponents(skeleton_map.astype(np.uint8), connectivity=8)

    # get TCL Instance
    all_pos_yxs = []
    if instance_count > 0:
        for instance_id in range(1, instance_count):
            pos_list = []
            ys, xs = np.where(instance_label_map == instance_id)
            pos_list = list(zip(ys, xs))

            ### FIX-ME, eliminate outlier
            if len(pos_list) < 3:
                continue
            
            pos_list_sorted = sort_and_expand_with_direction_v2(pos_list, f_direction, p_tcl_map)
            pos_list_sorted_with_id = add_id(pos_list_sorted, image_id=image_id)
            all_pos_yxs.append(pos_list_sorted_with_id)
    return all_pos_yxs






