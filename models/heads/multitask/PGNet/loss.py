from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.detection import DiceLoss


EPS = 1e-7
INF = 1e7


class PGLoss(nn.Module):

    def __init__(self, tcl_len, max_text_len, max_bbox_count, pad_num, **kwargs):
        super(PGLoss, self).__init__()
        self.tcl_len = tcl_len
        self.max_bbox_count = max_bbox_count
        self.max_text_len = max_text_len
        self.pad_num = pad_num
        self.center_loss = DiceLoss()

    def forward(self, predicts, labels):

        for k, v in predicts.items():
            print(k.ljust(19), v.size())
        for k, v in labels.items():
            print(k.ljust(19), v.size())

        # Unpack dictionaries
        tcl_maps       = labels['tcl_maps']
        border_maps    = labels['border_maps']
        direction_maps = labels['direction_maps']
        training_masks = labels['training_masks']
        label_list     = labels['label_list']
        pos_list       = labels['pos_list']
        pos_mask       = labels['pos_mask']

        f_center    = predicts['center']
        f_direction = predicts['direction']
        f_border    = predicts['border']
        f_character = predicts['character']
        
        # for all the batch_size
        pos_list, pos_mask, label_list, label_t = preprocess_pgloss(label_list, pos_list, pos_mask, 
                                                                    self.max_text_len, self.max_bbox_count, 
                                                                    self.pad_num, self.tcl_len)

        center_loss    =     self.center_loss(f_center   ,                 tcl_maps, training_masks)
        border_loss    =     self.border_loss(f_border   ,    border_maps, tcl_maps, training_masks)
        direction_loss =  self.direction_loss(f_direction, direction_maps, tcl_maps, training_masks)
        clssifier_loss = self.classifier_loss(f_character, pos_list, pos_mask, label_list, label_t)
        loss_total = center_loss + border_loss + direction_loss + 5 * clssifier_loss

        return {
                       'loss':           loss_total,
                "center_loss":    center_loss,
                "border_loss":    border_loss,
             "direction_loss": direction_loss,
            "classifier_loss": clssifier_loss,
        }

    def border_loss(self, f_border, l_border, l_score, l_mask):
        l_border_split, l_border_norm = torch.split(l_border, split_size_or_sections=[4, 1], dim=1)
        f_border_split = f_border

        b, c, h, w = l_border_norm.shape
        l_border_norm_split = l_border_norm.expand([b, 4 * c, h, w])

        b, c, h, w = l_score.shape
        l_border_score = l_score.expand([b, 4 * c, h, w])

        b, c, h, w = l_mask.shape
        l_border_mask = l_mask.expand([b, 4 * c, h, w])

        border_diff = l_border_split - f_border_split
        abs_border_diff = torch.abs(border_diff)
        border_sign = (abs_border_diff < 1.0).float().detach()
        border_in_loss = 0.5 * abs_border_diff * abs_border_diff * border_sign + \
                              (abs_border_diff - 0.5)   *   (1.0 - border_sign)
        border_out_loss = l_border_norm_split * border_in_loss
        border_loss = torch.sum(border_out_loss * l_border_score * l_border_mask) / \
                     (torch.sum(l_border_score * l_border_mask) + 1e-5)
        return border_loss

    def direction_loss(self, f_direction, l_direction, l_score, l_mask):
        l_direction_split, l_direction_norm = torch.split(l_direction, split_size_or_sections=[2, 1], dim=1)
        f_direction_split = f_direction

        b, c, h, w = l_direction_norm.shape
        l_direction_norm_split = l_direction_norm.expand([b, 2 * c, h, w])
        
        b, c, h, w = l_score.shape
        l_direction_score = l_score.expand([b, 2 * c, h, w])
        
        b, c, h, w = l_mask.shape
        l_direction_mask = l_mask.expand([b, 2 * c, h, w])
        
        direction_diff = l_direction_split - f_direction_split
        abs_direction_diff = torch.abs(direction_diff)
        direction_sign = (abs_direction_diff < 1.0).float().detach()
        direction_in_loss = 0.5 * abs_direction_diff * abs_direction_diff * direction_sign + \
                                 (abs_direction_diff - 0.5)     *    (1.0 - direction_sign)
        direction_out_loss = l_direction_norm_split * direction_in_loss
        direction_loss = torch.sum(direction_out_loss * l_direction_score * l_direction_mask) / \
                        (torch.sum(                     l_direction_score * l_direction_mask) + EPS)
        return direction_loss

    def classifier_loss(self, f_char, tcl_pos, tcl_mask, tcl_label, label_t):
        f_char = torch.transpose(f_char, [0, 2, 3, 1])
        tcl_pos = torch.reshape(tcl_pos, [-1, 3]).int()
        f_tcl_char = torch.gather(f_char, tcl_pos, dim=0)
        f_tcl_char = torch.reshape(f_tcl_char, [-1, 64, 37])  # len(Lexicon_Table) + 1
        f_tcl_char_fg, f_tcl_char_bg = torch.split(f_tcl_char, split_size_or_sections=[36, 1], dim=2)
        f_tcl_char_bg = f_tcl_char_bg * tcl_mask + (1.0 - tcl_mask) * 20.0
        
        b, c, l = tcl_mask.shape
        tcl_mask_fg = tcl_mask.expand([b, c, 36 * l]).detach()
        f_tcl_char_fg = f_tcl_char_fg * tcl_mask_fg + (1. - tcl_mask_fg) * (-20.)
        f_tcl_char_mask = torch.cat([f_tcl_char_fg, f_tcl_char_bg], dim=2)
        f_tcl_char_ld = torch.transpose(f_tcl_char_mask, (1, 0, 2))

        N, B, _ = f_tcl_char_ld.shape
        input_lengths = torch.tensor([N] * B, dtype='int64')
        cost = F.ctc_loss(log_probs=f_tcl_char_ld, input_lengths=input_lengths,
                            targets=  tcl_label,  target_lengths=label_t, blank=self.pad_num, reduction='none')
        return cost.mean()


def org_tcl_rois(batch_size, pos_lists, pos_masks, label_lists, tcl_len):

    pos_lists_, pos_masks_, label_lists_ = [], [], []
    img_bs = batch_size
    n_gpus = int(batch_size / img_bs)
    img_ids = np.array(pos_lists, dtype=np.int32)[:, 0, 0].copy()
    pos_lists_split, pos_masks_split, label_lists_split = [], [], []
    for i in range(n_gpus):
        pos_lists_split.append([])
        pos_masks_split.append([])
        label_lists_split.append([])

    for i in range(img_ids.shape[0]):
        img_id = img_ids[i]
        gpu_id = int(img_id / img_bs)
        img_id = img_id % img_bs
        pos_list = pos_lists[i].copy()
        pos_list[:, 0] = img_id
        pos_lists_split[gpu_id].append(pos_list)
        pos_masks_split[gpu_id].append(pos_masks[i].copy())
        label_lists_split[gpu_id].append(deepcopy(label_lists[i]))

    # repeat or delete
    for i in range(n_gpus):
        vp_len = len(pos_lists_split[i])
        if vp_len <= tcl_len:
            for j in range(0, tcl_len - vp_len):
                pos_list = pos_lists_split[i][j].copy()
                pos_mask = pos_masks_split[i][j].copy()
                label_list = deepcopy(label_lists_split[i][j])

                pos_masks_split[i].append(pos_mask)
                pos_lists_split[i].append(pos_list)
                label_lists_split[i].append(label_list)
        else:
            for j in range(0, vp_len - tcl_len):
                c_len = len(pos_lists_split[i])
                pop_id = np.random.permutation(c_len)[0]
                pos_lists_split[i].pop(pop_id)
                pos_masks_split[i].pop(pop_id)
                label_lists_split[i].pop(pop_id)
    # merge
    for i in range(n_gpus):
        pos_lists_.extend(pos_lists_split[i])
        pos_masks_.extend(pos_masks_split[i])
        label_lists_.extend(label_lists_split[i])

    return pos_lists_, pos_masks_, label_lists_


def preprocess_pgloss(label_list, pos_list, pos_mask, max_text_len, max_bbox_count, pad_num, tcl_len):
    label_list = label_list.numpy()
    B, _, _, _ = label_list.shape
    pos_list = pos_list.numpy()
    pos_mask = pos_mask.numpy()
    pos_list_t = []
    pos_mask_t = []
    label_list_t = []
    for i in range(B):
        for j in range(max_bbox_count):
            if pos_mask[i, j].any():
                pos_list_t.append(pos_list[i][j])
                pos_mask_t.append(pos_mask[i][j])
                label_list_t.append(label_list[i][j])
    pos_list, pos_mask, label_list = org_tcl_rois(B, pos_list_t, pos_mask_t, label_list_t, tcl_len)

    label = []
    tt = [l.tolist() for l in label_list]
    for i in range(tcl_len):
        k = 0
        for j in range(max_text_len):
            if tt[i][j][0] != pad_num:
                k += 1
            else:
                break
        label.append(k)
    label = torch.tensor(label).int()
    pos_list = torch.tensor(np.array(pos_list))
    pos_mask = torch.tensor(np.array(pos_mask))
    label_list = torch.squeeze(
                torch.tensor(np.array(label_list)), dim=2).int()

    return pos_list, pos_mask, label_list, label


