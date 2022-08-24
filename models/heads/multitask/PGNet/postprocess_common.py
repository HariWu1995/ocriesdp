from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from itertools import groupby
from skimage.morphology._skeletonize import thin


def get_dict(character_dict_path):
    character_str = ""
    with open(character_dict_path, "rb") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.decode('utf-8').strip("\n").strip("\r\n")
            character_str += line
        dict_character = list(character_str)
    return dict_character


def get_keep_pos_idxs(labels, remove_blank=None):
    """
    Remove duplicate and get pos idxs of keep items.
    The value of keep_blank should be [None, 95].
    """
    duplicate_len_list = []
    keep_pos_idx_list = []
    keep_char_idx_list = []
    for k, v_ in groupby(labels):
        current_len = len(list(v_))
        if k != remove_blank:
            current_idx = int(sum(duplicate_len_list) + current_len // 2)
            keep_pos_idx_list.append(current_idx)
            keep_char_idx_list.append(k)
        duplicate_len_list.append(current_len)
    return keep_char_idx_list, keep_pos_idx_list


def softmax(logits):
    """
    logits: N x d
    """
    max_value = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - max_value)
    exp_sum = np.sum(exp, axis=1, keepdims=True)
    dist = exp / exp_sum
    return dist


def remove_blank(labels, blank=0):
    new_labels = [x for x in labels if x != blank]
    return new_labels


def insert_blank(labels, blank=0):
    new_labels = [blank]
    for l in labels:
        new_labels += [l, blank]
    return new_labels


def ctc_greedy_decoder(probs_seq, blank=95, keep_blank_in_idxs=True):
    """
    CTC greedy (best path) decoder.
    """
    raw_str = np.argmax(np.array(probs_seq), axis=1)
    remove_blank_in_pos = None if keep_blank_in_idxs else blank
    dedup_str, keep_idx_list = get_keep_pos_idxs(raw_str, remove_blank=remove_blank_in_pos)
    dst_str = remove_blank(dedup_str, blank=blank)
    return dst_str, keep_idx_list


def sort_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    def sort_part_with_direction(pos_list, point_direction):
        pos_list = np.array(pos_list).reshape(-1, 2)
        point_direction = np.array(point_direction).reshape(-1, 2)
        average_direction = np.mean(point_direction, axis=0, keepdims=True)
        pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
        sorted_list = pos_list[np.argsort(pos_proj_leng)].tolist()
        sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
        return sorted_list, sorted_direction

    pos_list = np.array(pos_list).reshape(-1, 2)
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # x, y
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    sorted_point, sorted_direction = sort_part_with_direction(pos_list, point_direction)

    point_num = len(sorted_point)
    if point_num >= 16:
        middle_num = point_num // 2
        first_part_point      = sorted_point[    :middle_num]
        first_point_direction = sorted_direction[:middle_num]
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(first_part_point, first_point_direction)

        last_part_point      = sorted_point[    middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(last_part_point, last_point_direction)
        sorted_point     = sorted_fist_part_point     + sorted_last_part_point
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

    return sorted_point, np.array(sorted_direction)


def add_id(pos_list, image_id=0):
    """
    Add id for gather feature, for inference.
    """
    new_list = []
    for item in pos_list:
        new_list.append((image_id, item[0], item[1]))
    return new_list


def sort_and_expand_with_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    h, w, _ = f_direction.shape
    sorted_list, point_direction = sort_with_direction(pos_list, f_direction)

    # expand along
    point_num = len(sorted_list)
    sub_direction_len = max(point_num // 3, 2)
    left_direction = point_direction[         :sub_direction_len , :]
    right_dirction = point_direction[point_num-sub_direction_len:, :]

    left_average_direction = -np.mean(left_direction, axis=0, keepdims=True)
    left_average_len = np.linalg.norm(left_average_direction)
    left_start = np.array(sorted_list[0])
    left_step = left_average_direction / (left_average_len + 1e-6)

    right_average_direction = np.mean(right_dirction, axis=0, keepdims=True)
    right_average_len = np.linalg.norm(right_average_direction)
    right_step = right_average_direction / (right_average_len + 1e-6)
    right_start = np.array(sorted_list[-1])

    append_num = max(int((left_average_len + right_average_len) / 2.0 * 0.15), 1)
    left_list = []
    right_list = []
    for i in range(append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype('int32').tolist()
        if ly < h and lx < w and (ly, lx) not in left_list:
            left_list.append((ly, lx))
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype('int32').tolist()
        if ry < h and rx < w and (ry, rx) not in right_list:
            right_list.append((ry, rx))

    all_list = left_list[::-1] + sorted_list + right_list
    return all_list


def sort_and_expand_with_direction_v2(pos_list, f_direction, binary_tcl_map):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    binary_tcl_map: h x w
    """
    h, w, _ = f_direction.shape
    sorted_list, point_direction = sort_with_direction(pos_list, f_direction)

    # expand along
    point_num = len(sorted_list)
    sub_direction_len = max(point_num // 3, 2)
    left_direction = point_direction[         :sub_direction_len , :]
    right_dirction = point_direction[point_num-sub_direction_len:, :]

    left_average_direction = -np.mean(left_direction, axis=0, keepdims=True)
    left_average_len = np.linalg.norm(left_average_direction)
    left_start = np.array(sorted_list[0])
    left_step = left_average_direction / (left_average_len + 1e-6)

    right_average_direction = np.mean(right_dirction, axis=0, keepdims=True)
    right_average_len = np.linalg.norm(right_average_direction)
    right_step = right_average_direction / (right_average_len + 1e-6)
    right_start = np.array(sorted_list[-1])

    append_num = max(int((left_average_len + right_average_len) / 2.0 * 0.15), 1)
    max_append_num = 2 * append_num

    left_list = []
    right_list = []
    for i in range(max_append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).flatten().astype('int32').tolist()
        if ly < h and lx < w and (ly, lx) not in left_list:
            if binary_tcl_map[ly, lx] > 0.5:
                left_list.append((ly, lx))
            else:
                break

    for i in range(max_append_num):
        ry, rx = np.round(right_start + right_step * (i + 1)).flatten().astype('int32').tolist()
        if ry < h and rx < w and (ry, rx) not in right_list:
            if binary_tcl_map[ry, rx] > 0.5:
                right_list.append((ry, rx))
            else:
                break

    all_list = left_list[::-1] + sorted_list + right_list
    return all_list


# for refine module
def extract_main_direction(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    pos_list = np.array(pos_list)
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    average_direction = average_direction / (np.linalg.norm(average_direction) + 1e-6)
    return average_direction


def sort_by_direction_with_image_id_deprecated(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[id, y, x], [id, y, x], [id, y, x] ...]
    """
    pos_list_full = np.array(pos_list).reshape(-1, 3)
    pos_list = pos_list_full[:, 1:]
    point_direction = f_direction[pos_list[:, 0], pos_list[:, 1]]  # x, y
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    average_direction = np.mean(point_direction, axis=0, keepdims=True)
    pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
    sorted_list = pos_list_full[np.argsort(pos_proj_leng)].tolist()
    return sorted_list


def sort_by_direction_with_image_id(pos_list, f_direction):
    """
    f_direction: h x w x 2
    pos_list: [[y, x], [y, x], [y, x] ...]
    """
    def sort_part_with_direction(pos_list_full, point_direction):
        pos_list_full = np.array(pos_list_full).reshape(-1, 3)
        pos_list = pos_list_full[:, 1:]
        point_direction = np.array(point_direction).reshape(-1, 2)
        average_direction = np.mean(point_direction, axis=0, keepdims=True)
        pos_proj_leng = np.sum(pos_list * average_direction, axis=1)
        sorted_list = pos_list_full[np.argsort(pos_proj_leng)].tolist()
        sorted_direction = point_direction[np.argsort(pos_proj_leng)].tolist()
        return sorted_list, sorted_direction

    pos_list = np.array(pos_list).reshape(-1, 3)
    point_direction = f_direction[pos_list[:, 1], pos_list[:, 2]]  # x, y
    point_direction = point_direction[:, ::-1]  # x, y -> y, x
    sorted_point, sorted_direction = sort_part_with_direction(pos_list, point_direction)

    point_num = len(sorted_point)
    if point_num >= 16:
        middle_num = point_num // 2
        first_part_point = sorted_point[:middle_num]
        first_point_direction = sorted_direction[:middle_num]
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(first_part_point, first_point_direction)

        last_part_point = sorted_point[middle_num:]
        last_point_direction = sorted_direction[middle_num:]
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(last_part_point, last_point_direction)
        sorted_point = sorted_fist_part_point + sorted_last_part_point
        sorted_direction = sorted_fist_part_direction + sorted_last_part_direction

    return sorted_point







