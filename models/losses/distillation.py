
import cv2
import numpy as np
import torch
import torch.nn as nn

from models.losses.common import DMLLoss, DistanceLoss
from models.heads.detection.DB.loss import DBLoss
from models.heads.recognition.CTC.loss import CTCLoss
from models.heads.recognition.SAR.loss import SARLoss


def aggregate_loss(loss_dict):
    if "loss" in loss_dict.keys():
        return loss_dict
    else:
        loss_dict["loss"] = 0.
        for k, value in loss_dict.items():
            if k == "loss":
                continue
            else:
                loss_dict["loss"] += value
        return loss_dict


class DistilledDMLLoss(DMLLoss):

    def __init__(self, model_name_pairs=[], act=None, use_log=False,
                    key=None, multi_head=False, dis_head='ctc', maps_name=None, name="dml"):
        super().__init__(act=act, use_log=use_log)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.multi_head = multi_head
        self.dis_head = dis_head
        self.model_name_pairs = self._check_model_name_pairs(model_name_pairs)
        self.name = name
        self.maps_name = self._check_maps_name(maps_name)

    def _check_model_name_pairs(self, model_name_pairs):
        if not isinstance(model_name_pairs, list):
            return []
        elif isinstance(model_name_pairs[0], list) and isinstance(model_name_pairs[0][0], str):
            return model_name_pairs
        else:
            return [model_name_pairs]

    def _check_maps_name(self, maps_name):
        if maps_name is None:
            return None
        elif type(maps_name) == str:
            return [maps_name]
        elif type(maps_name) == list:
            return [maps_name]
        else:
            return None

    def _slice_out(self, outs):
        new_outs = {}
        for k in self.maps_name:
            if k == "thrink_maps":
                new_outs[k] = outs[:, 0, :, :]
            elif k == "threshold_maps":
                new_outs[k] = outs[:, 1, :, :]
            elif k == "binary_maps":
                new_outs[k] = outs[:, 2, :, :]
            else:
                continue
        return new_outs

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]

            if self.maps_name is None:
                if self.multi_head:
                    loss = super().forward(out1[self.dis_head], out2[self.dis_head])
                else:
                    loss = super().forward(out1, out2)
                if isinstance(loss, dict):
                    for key in loss:
                        loss_dict["{}_{}_{}_{}".format(key, pair[0], pair[1], idx)] = loss[key]
                else:
                    loss_dict["{}_{}".format(self.name, idx)] = loss
            else:
                outs1 = self._slice_out(out1)
                outs2 = self._slice_out(out2)
                for _c, k in enumerate(outs1.keys()):
                    loss = super().forward(outs1[k], outs2[k])
                    if isinstance(loss, dict):
                        for key in loss:
                            loss_dict["{}_{}_{}_{}_{}".format(key, pair[0], pair[1], self.maps_name, idx)] = loss[key]
                    else:
                        loss_dict["{}_{}_{}".format(self.name, self.maps_name[_c], idx)] = loss

        loss_dict = aggregate_loss(loss_dict)

        return loss_dict


class DistilledCTCLoss(CTCLoss):

    def __init__(self, model_name_list=[], key=None, multi_head=False, name="loss_ctc"):
        super().__init__()
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            if self.multi_head:
                assert 'ctc' in out, 'multi head has multi out'
                loss = super().forward(out['ctc'], batch[:2] + batch[3:])
            else:
                loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, model_name, idx)] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        return loss_dict


class DistilledSARLoss(SARLoss):

    def __init__(self, model_name_list=[], key=None, multi_head=False, name="loss_sar", **kwargs):
        ignore_index = kwargs.get('ignore_index', 92)
        super().__init__(ignore_index=ignore_index)
        self.model_name_list = model_name_list
        self.key = key
        self.name = name
        self.multi_head = multi_head

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            if self.multi_head:
                assert 'sar' in out, 'multi head has multi out'
                loss = super().forward(out['sar'], batch[:1] + batch[2:])
            else:
                loss = super().forward(out, batch)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, model_name, idx)] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss
        return loss_dict


class DistilledDBLoss(DBLoss):

    def __init__(self, model_name_list=[], balance_loss=True, loss_type='DiceLoss', 
                       alpha=5, beta=10, ohem_ratio=3, name="db", **kwargs):
        super().__init__()
        self.model_name_list = model_name_list
        self.name = name
        self.key = None

    def forward(self, predicts, batch):
        loss_dict = {}
        for idx, model_name in enumerate(self.model_name_list):
            out = predicts[model_name]
            if self.key is not None:
                out = out[self.key]
            loss = super().forward(out, batch)

            if isinstance(loss, dict):
                for key in loss.keys():
                    if key == "loss":
                        continue
                    name = "{}_{}_{}".format(self.name, model_name, key)
                    loss_dict[name] = loss[key]
            else:
                loss_dict["{}_{}".format(self.name, model_name)] = loss

        loss_dict = aggregate_loss(loss_dict)
        return loss_dict


class DistilledDilatedDBLoss(DBLoss):

    def __init__(self, model_name_pairs=[], key=None, balance_loss=True, loss_type='DiceLoss',
                        alpha=5, beta=10, ohem_ratio=3, name="dilated_dbloss"):
        super().__init__()
        self.model_name_pairs = model_name_pairs
        self.name = name
        self.key = key

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            stu_outs = predicts[pair[0]]
            tch_outs = predicts[pair[1]]
            if self.key is not None:
                stu_preds = stu_outs[self.key]
                tch_preds = tch_outs[self.key]

            stu_shrink_maps = stu_preds[:, 0, :, :]
            stu_binary_maps = stu_preds[:, 2, :, :]

            # dilation to teacher prediction
            dilation_w = np.array([[1, 1], [1, 1]])
            th_shrink_maps = tch_preds[:, 0, :, :]
            th_shrink_maps = th_shrink_maps.numpy() > 0.3  # thresh = 0.3 
            dilate_maps = np.zeros_like(th_shrink_maps).astype(np.float32)
            for i in range(th_shrink_maps.shape[0]):
                dilate_maps[i] = cv2.dilate(th_shrink_maps[i,:,:].astype(np.uint8), dilation_w)
            th_shrink_maps = torch.tensor(dilate_maps)

            label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = batch[1:]

            # calculate the shrink map loss
            bce_loss = self.alpha * self.bce_loss(stu_shrink_maps, th_shrink_maps, label_shrink_mask)
            loss_binary_maps     = self.dice_loss(stu_binary_maps, th_shrink_maps, label_shrink_mask)

            loss_dict["{}_{}_{}".format(self.name, pair[0], pair[1])] = bce_loss + loss_binary_maps

        loss_dict = aggregate_loss(loss_dict)
        return loss_dict


class DistilledDistanceLoss(DistanceLoss):

    def __init__(self, mode="l2", model_name_pairs=[], key=None, name="loss_distance", **kargs):
        super().__init__(mode=mode, **kargs)
        assert isinstance(model_name_pairs, list)
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.name = name + "_l2"

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = super().forward(out1, out2)
            if isinstance(loss, dict):
                for key in loss:
                    loss_dict["{}_{}_{}".format(self.name, key, idx)] = loss[key]
            else:
                loss_dict["{}_{}_{}_{}".format(self.name, pair[0], pair[1], idx)] = loss
        return loss_dict


