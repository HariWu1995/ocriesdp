from pathlib import Path
from PIL import Image

import math
import cv2
import numpy as np
import pandas as pd
import torch

from src.data.datasets import infer_statistics
from src.data.label_multi import OCRLabelEncoder as LabelEncoder


EPS = 1e-7
CHANNEL_SCALER = {
    2: { 'mean' : 0.485, 'std' : 0.229, },
    1: { 'mean' : 0.456, 'std' : 0.224, },
    0: { 'mean' : 0.406, 'std' : 0.225, },
}


class Dataloader:

    def __init__(self, image_df: pd.DataFrame, bbox_df: pd.DataFrame,
                    config_path: str or Path, processor_config: dict = {},
                    batch_size: int = 16, shuffle: bool = False, ):
        image_df = image_df.drop_duplicates(subset=['image_id'])
        self.image_df = image_df.set_index(keys=['image_id'])
        self.bbox_df = bbox_df.set_index(keys=['image_id'])
        self.indices = list(self.image_df.index)
        self.shuffle = shuffle
        self.batch_size = batch_size

        processor_config.update(dict(batch_size=batch_size))
        self.Preprocessor = PGProcessor(config_path, **processor_config)
        self.LabelEncoder = LabelEncoder(encode_text=True, use_space_char=True,
                                        max_text_len=self.Preprocessor.max_text_len, 
                                      character_list=self.Preprocessor.vocabulary, 
                                          label_attr='label', bbox_attr='polygons', text_attr='text')

    def __len__(self):
        return math.ceil(self.image_df.shape[0] / self.batch_size)

    def __getitem__(self, index):
        max_index = len(self)
        assert index < max_index, f"index = {index} exceeds {max_index}"
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        data = []
        for image_id in self.indices[start_index:end_index]:
            image_data = {
                'image' : Image.open(self.image_df.loc[image_id, 'image_path']),
                'label' : { column : self.bbox_df.loc[image_id, column].values.tolist() 
                        for column in ['polygons', 'text'] },
            }
            image_data = self.LabelEncoder(image_data)
            image_data = self.Preprocessor(image_data)
            if image_data is None:
                continue
            
            data.append({k: v for k, v in image_data.items()})

        # Convert to tensor
        batch_df = pd.DataFrame.from_dict(data)
        
        data_ts = dict()
        for col in ['image', 'tcl_maps', 'tcl_label_maps', 'border_maps', 'direction_maps', 
                    'training_masks', 'label_list', 'pos_list', 'pos_mask',]:
            col_arr = np.stack(batch_df[col].values, axis=3).transpose([3,0,1,2])
            col_ts = torch.tensor(col_arr, dtype=torch.float32)
            data_ts[col] = col_ts
        
        return data_ts

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices) # inplace operation


class PGProcessor(object):

    scales = [1.] # [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.0, 1.0, 1.0]

    def __init__(self, config_path, tcl_len: int = 64, batch_size: int = 4, 
                                input_size: int = 512, min_crop_size: int = 24, **kwargs):
        self.tcl_len = tcl_len
        self.input_size = input_size
        self.batch_size = batch_size
        self.min_crop_size = min_crop_size

        config = infer_statistics(config_path)
        config.update(kwargs) # overwrite

        self.image_size      = config['image_size']
        self.max_bbox_count  = config['max_bbox_count']
        self.max_bbox_size   = config['max_bbox_size']
        self.min_bbox_size   = config['min_bbox_size']
        self.max_text_len    = config['max_text_len']
        self.vocabulary = list(config['vocab'])
        self.vocab_size =  len(config['vocab'])
        self.image_id = 0

    def __call__(self, data):
        input_size = self.input_size
        image      = data['image'].convert('RGB')
        text_polys = data['polys']
        text_tags  = data['ignore_tags']
        text_strs  = data['texts']

        # Padding
        W, H = image.size
        newsize_image = np.max([W, H, self.image_size // 2])
        padded_image = Image.new(image.mode, (newsize_image, newsize_image), (0, 0, 0))
        padded_image.paste(image, (0, 0))
        
        image = np.array(padded_image)
        h, w = image.shape[:2]
        text_polys, text_tags, hv_tags = self.check_and_validate_polys(text_polys, text_tags, (h, w))
        if text_polys.shape[0] <= 0:
            return None

        # set aspect ratio and keep area fix
        asp_scales = np.arange(1.0, 1.55, 0.1)
        asp_scale = np.random.choice(asp_scales)
        if np.random.rand() < 0.5:
            asp_scale = 1.0 / asp_scale
        asp_scale = math.sqrt(asp_scale)

        asp_wx = asp_scale
        asp_hy = 1.0 / asp_scale
        image = cv2.resize(image, dsize=None, fx=asp_wx, fy=asp_hy)
        text_polys[:, :, 0] *= asp_wx
        text_polys[:, :, 1] *= asp_hy

        h, w, _ = image.shape
        if max(h, w) > 2048:
            rd_scale = 2048. / max(h, w)
            image = cv2.resize(image, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale
        h, w, _ = image.shape
        if min(h, w) < 16:
            print(f"Size is too small, h = {h}, w = {w}")
            return None

        # no background
        # image, text_polys, text_tags, hv_tags, text_strs = self.crop_area(
        # image, text_polys, text_tags, hv_tags, text_strs, crop_background=False)

        if text_polys.shape[0] == 0:
            print(f"Image contains no polygons!")
            return None

        # continue for all ignore case
        if np.sum((text_tags * 1.0)) >= text_tags.size:
            print(f"Image contains no text!")
            return None

        new_h, new_w, _ = image.shape
        if (new_h is None) or (new_w is None):
            print(f"Crop size is too small, h = {new_h}, w = {new_w}")
            return None

        # resize image
        std_ratio = float(input_size) / max(new_w, new_h)
        rz_scale = std_ratio * np.random.choice(self.scales)
        image = cv2.resize(image, dsize=None, fx=rz_scale, fy=rz_scale)
        text_polys[:, :, 0] *= rz_scale
        text_polys[:, :, 1] *= rz_scale

        # add gaussian blur
        if np.random.rand() < 0.1 * 0.5:
            ks = np.random.permutation(5)[0] + 1
            ks = int(ks / 2) * 2 + 1
            image = cv2.GaussianBlur(image, ksize=(ks, ks), sigmaX=0, sigmaY=0)

        # add brighter
        if np.random.rand() < 0.1 * 0.5:
            image = image * (1.0 + np.random.rand() * 0.5)
            image = np.clip(image, 0.0, 255.0)

        # add darker
        if np.random.rand() < 0.1 * 0.5:
            image = image * (1.0 - np.random.rand() * 0.5)
            image = np.clip(image, 0.0, 255.0)

        # Padding the im to [input_size, input_size]
        new_h, new_w, _ = image.shape
        if min(new_w, new_h) < input_size * 0.5:
            print(f"Re-Size is too small, h = {new_h}, w = {new_w}")
            return None
        image_padded = np.ones((input_size, input_size, 3), dtype=np.float32)
        for channel, channel_scale in CHANNEL_SCALER.items():
            image_padded[:, :, channel] = channel_scale['mean'] * 255.

        # Random the start position
        del_h = input_size - new_h
        del_w = input_size - new_w
        sh, sw = 0, 0
        if del_h > 1:
            sh = int(np.random.rand() * del_h)
        if del_w > 1:
            sw = int(np.random.rand() * del_w)

        # Padding
        image_padded[sh:sh+new_h, sw:sw+new_w, :] = image.copy()
        text_polys[:, :, 0] += sw
        text_polys[:, :, 1] += sh

        score_map, score_label_map, border_map, direction_map, \
            training_mask, pos_list, pos_mask, label_list, _ = self.generate_tcl_ctc_label(input_size, input_size,
                                                                                           text_polys, text_tags, text_strs, 0.05)
        if len(label_list) == 0:  # eliminate negative samples
            print(f"Length of labels is 0")
            return None
        pos_list_temp = np.zeros([64, 3])
        pos_mask_temp = np.zeros([64, 1])
        label_list_temp = np.zeros([self.max_text_len, 1]) + self.vocab_size

        for i, label in enumerate(label_list):
            n = len(label)
            if n > self.max_text_len:
                label_list[i] = label[:self.max_text_len]
                continue
            while n < self.max_text_len:
                label.append([self.vocab_size])
                n += 1

        for i in range(len(label_list)):
            label_list[i] = np.array(label_list[i]).reshape([self.max_text_len, 1])

        if len(pos_list) == 0:
            print(f"Length of labels is 0")
            return None
        if len(pos_list) > self.max_bbox_count:
            print(f"Length of labels is too high, {len(pos_list)} > {self.max_bbox_count}")
            return None

        for __ in range(self.max_bbox_count - len(pos_list), 0, -1):
            pos_list.append(pos_list_temp)
            pos_mask.append(pos_mask_temp)
            label_list.append(label_list_temp)

        if  self.image_id == self.batch_size - 1:
            self.image_id = 0
        else:
            self.image_id += 1

        for channel, channel_scale in CHANNEL_SCALER.items():
            image_padded[:, :, channel] -= 255. * channel_scale['mean']
            image_padded[:, :, channel] /= 255. * channel_scale['std']

        image_padded = image_padded.transpose((2, 0, 1))
        image_padded = image_padded[::-1, :, :]

        training_masks =   training_mask[np.newaxis, :, :]
        tcl_maps       =       score_map[np.newaxis, :, :]
        tcl_label_maps = score_label_map[np.newaxis, :, :]
        border_maps    =      border_map.transpose((2, 0, 1))
        direction_maps =   direction_map.transpose((2, 0, 1))

        pos_mask   = np.array(  pos_mask)
        pos_list   = np.array(  pos_list)
        label_list = np.array(label_list)

        data.update({
                     'image' : image_padded,
                  'tcl_maps' : tcl_maps,
            'tcl_label_maps' : tcl_label_maps,
               'border_maps' :    border_maps,
            'direction_maps' : direction_maps,
            'training_masks' : training_masks,
                'label_list' : label_list,
                  'pos_list' :   pos_list,
                  'pos_mask' :   pos_mask,
        })
        return data

    def quad_area(self, poly):
        """
        compute area of a polygon
        """
        edge = [(poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
                (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
                (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
                (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1]),]
        return np.sum(edge) / 2.

    def gen_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
        box = np.array(cv2.boxPoints(rect))

        first_point_idx = 0
        min_dist = 1e4
        for i in range(4):
            dist = np.linalg.norm(box[(i + 0) % 4] - poly[0                 ]) + \
                   np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                   np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2    ]) + \
                   np.linalg.norm(box[(i + 3) % 4] - poly[               - 1])
            if dist < min_dist:
                min_dist = dist
                first_point_idx = i
        for i in range(4):
            min_area_quad[i] = box[(first_point_idx + i) % 4]

        return min_area_quad

    def check_and_validate_polys(self, polys, tags, im_size):
        """
        check if text poly is in the same direction, and filter invalid polygons
        """
        h, w = im_size
        if polys.shape[0] == 0:
            return polys, np.array([]), np.array([])
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

        validated_polys = []
        validated_tags = []
        hv_tags = []
        for poly, tag in zip(polys, tags):
            quad = self.gen_quad_from_poly(poly)
            p_area = self.quad_area(quad)
            if abs(p_area) < 1:
                print('invalid poly')
                continue
            if p_area > 0:
                if tag == False:
                    print('poly in wrong direction')
                    tag = True  # reversed cases should be ignore
                poly = poly[tuple([0] + list(range(15, 0, -1))), :]
                quad = quad[tuple([0] + list(range( 3, 0, -1))), :]

            len_w = np.linalg.norm(quad[0] - quad[1]) + np.linalg.norm(quad[3] - quad[2])
            len_h = np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[1] - quad[2])
            hv_tag = 1

            if len_w * 2.0 < len_h:
                hv_tag = 0

            validated_polys.append(poly)
            validated_tags.append(tag)
            hv_tags.append(hv_tag)

        return np.array(validated_polys), np.array(validated_tags), np.array(hv_tags)

    def crop_area(self, image, polys, tags, hv_tags, txts, crop_background=False, max_tries=25):
        """
        make random crop from the input image
        """
        h, w, _ = image.shape
        pad_h = h // 10
        pad_w = w // 10
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1
            w_array[minx + pad_w:maxx + pad_w] = 1

        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        if len(h_axis) == 0 or len(w_axis) == 0:
            return image, polys, tags, hv_tags, txts

        for i in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            yy = np.random.choice(h_axis, size=2)

            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h

            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin < self.min_crop_size) or \
               (ymax - ymin < self.min_crop_size):
                continue
            if polys.shape[0] != 0:
                poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                  & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
                selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
            else:
                selected_polys = []
            if len(selected_polys) == 0:
                # no text in this area
                if crop_background:
                    txts_tmp = []
                    for selected_poly in selected_polys:
                        txts_tmp.append(txts[selected_poly])
                    txts = txts_tmp
                    return image[ymin:ymax+1, xmin:xmax+1, :], polys[selected_polys], \
                                                                tags[selected_polys], \
                                                             hv_tags[selected_polys], txts
                else:
                    continue

            image = image[ymin:ymax+1, xmin:xmax+1, :]
            polys   =   polys[selected_polys]
            tags    =    tags[selected_polys]
            hv_tags = hv_tags[selected_polys]
            txts_tmp = []
            for selected_poly in selected_polys:
                txts_tmp.append(txts[selected_poly])
            txts = txts_tmp
            polys[:, :, 0] -= xmin
            polys[:, :, 1] -= ymin
            return image, polys, tags, hv_tags, txts

        return image, polys, tags, hv_tags, txts

    def fit_and_gather_tcl_points_v2(self, min_area_quad, poly, max_h, max_w,
                                        fixed_point_num=64, image_id=0, reference_height=3):
        """
        Find the center point of poly as key_points, then fit and gather.
        """
        key_point_xys = []
        point_num = poly.shape[0]
        for idx in range(point_num // 2):
            center_point = (poly[idx] + poly[point_num - 1 - idx]) / 2.0
            key_point_xys.append(center_point)

        tmp_image = np.zeros(shape=(max_h, max_w,), dtype='float32')
        cv2.polylines(tmp_image, [np.array(key_point_xys).astype('int32')], False, 1.0)
        ys, xs = np.where(tmp_image > 0)
        xy_text = np.array(list(zip(xs, ys)), dtype='float32')

        left_center_pt  = ((min_area_quad[0] - min_area_quad[1]) / 2.0).reshape(1, 2)
        right_center_pt = ((min_area_quad[1] - min_area_quad[2]) / 2.0).reshape(1, 2)

        proj_unit_vec = (right_center_pt - left_center_pt) / (np.linalg.norm(
                         right_center_pt - left_center_pt) + EPS)
        proj_unit_vec_tile  = np.tile( proj_unit_vec, (xy_text.shape[0], 1))  # (n, 2)
        left_center_pt_tile = np.tile(left_center_pt, (xy_text.shape[0], 1))  # (n, 2)

        xy_text_to_left_center = xy_text - left_center_pt_tile
        proj_value = np.sum(xy_text_to_left_center * proj_unit_vec_tile, axis=1)
        xy_text = xy_text[np.argsort(proj_value)]

        # convert to np and keep the num of point not greater then fixed_point_num
        pos_info = np.array(xy_text).reshape(-1, 2)[:, ::-1]  # xy-> yx
        point_num = len(pos_info)
        if point_num > fixed_point_num:
            keep_ids = [int((point_num * 1.0 / fixed_point_num) * x) for x in range(fixed_point_num)]
            pos_info = pos_info[keep_ids, :]

        keep = int(min(len(pos_info), fixed_point_num))
        if np.random.rand() < 0.2 and reference_height >= 3:
            dl = (np.random.rand(keep) - 0.5) * reference_height * 0.3
            random_float = np.array([1, 0]).reshape([1, 2]) * dl.reshape([keep, 1])

            pos_info += random_float
            pos_info[:, 0] = np.clip(pos_info[:, 0], 0, max_h - 1)
            pos_info[:, 1] = np.clip(pos_info[:, 1], 0, max_w - 1)

        # padding to fixed length
        pos_l      = np.zeros((self.tcl_len, 3), dtype=np.int32)
        pos_l[:, 0] = np.ones((self.tcl_len, )) * image_id
        pos_m      = np.zeros((self.tcl_len, 1), dtype=np.float32)
        pos_l[:keep, 1:] = np.round(pos_info).astype(np.int32)
        pos_m[:keep] = 1.0
        return pos_l, pos_m

    def generate_direction_map(self, poly_quads, n_char, direction_map):
        width_list = []
        height_list = []
        for quad in poly_quads:
            quad_w = (np.linalg.norm(quad[0] - quad[1]) +
                      np.linalg.norm(quad[2] - quad[3])) / 2.0
            quad_h = (np.linalg.norm(quad[0] - quad[3]) +
                      np.linalg.norm(quad[2] - quad[1])) / 2.0
            width_list.append(quad_w)
            height_list.append(quad_h)

        norm_width      = max(sum(width_list) /          n_char , 1.0)
        average_height = max(sum(height_list) / len(height_list), 1.0)
        k = 1
        for quad in poly_quads:
            direct_vector_full = ((quad[1] + quad[2]) - (quad[0] + quad[3])) / 2.0
            direct_vector = direct_vector_full / (np.linalg.norm(direct_vector_full) + EPS) * norm_width
            direction_label = tuple(
                map(float, [direct_vector[0], direct_vector[1], 1.0 / average_height]))
            cv2.fillPoly(direction_map, quad.round().astype(np.int32)[np.newaxis, :, :],
                         direction_label)
            k += 1
        return direction_map

    def calculate_average_height(self, poly_quads):
        """
        """
        height_list = []
        for quad in poly_quads:
            quad_h = (np.linalg.norm(quad[0] - quad[3]) + np.linalg.norm(quad[2] - quad[1])) / 2.0
            height_list.append(quad_h)
        average_height = max(sum(height_list) / len(height_list), 1.0)
        return average_height

    def generate_tcl_ctc_label(self, h, w, polys, tags, text_strs, ds_ratio, tcl_ratio=0.3, shrink_ratio_of_width=0.15):
        """
        Generate polygon.
        """
        h, w = int(h * ds_ratio), int(w * ds_ratio)
        polys = polys * ds_ratio

        score_map_big   = np.zeros([h, w   ], dtype=np.float32)
        score_map       = np.zeros([h, w   ], dtype=np.float32)
        score_label_map = np.zeros([h, w   ], dtype=np.float32)
        tbo_map         = np.zeros([h, w, 5], dtype=np.float32)
        training_mask   =  np.ones([h, w   ], dtype=np.float32)
        direction_map   =  np.ones([h, w, 3]) \
                        * np.array([0, 0, 1], dtype=np.float32).reshape([1, 1, 3])

        label_idx = 0
        score_label_map_text_label_list = []
        pos_list, pos_mask, label_list = [], [], []
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            poly = poly_tag[0]
            tag = poly_tag[1]

            # generate min_area_quad
            min_area_quad, center_point = self.gen_min_area_quad_from_poly(poly)
            min_area_quad_h = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[3]) +
                                     np.linalg.norm(min_area_quad[1] - min_area_quad[2]))
            min_area_quad_w = 0.5 * (np.linalg.norm(min_area_quad[0] - min_area_quad[1]) +
                                     np.linalg.norm(min_area_quad[2] - min_area_quad[3]))

            min_area_squad = min(min_area_quad_h, min_area_quad_w)
            if min_area_squad < self.min_bbox_size * ds_ratio \
            or min_area_squad > self.max_bbox_size * ds_ratio:
                print(f"Min size of quad area {min_area_squad} is too small (< {self.min_bbox_size * ds_ratio}) " +
                                                               f"or too big (> {self.max_bbox_size * ds_ratio})!")
                continue

            if tag:
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0.15)
            else:
                text_label = self.prepare_text_label(text_strs[poly_idx])
                text_label_index_list = text_label 
                # text_label_index_list = [[self.vocabulary.index(c_)] for c_ in text_label if c_ in self.vocabulary]
                if len(text_label_index_list) == 0:
                    print(f"All characters in {text_label} are out-of-vocab")
                    continue

                tcl_poly = self.poly2tcl(poly, tcl_ratio)
                tcl_quads = self.poly2quads(tcl_poly)
                poly_quads = self.poly2quads(poly)

                stcl_quads, quad_index = self.shrink_poly_along_width(tcl_quads,
                                                                      shrink_ratio_of_width=shrink_ratio_of_width,
                                                                      expand_height_ratio=1.0 / tcl_ratio)

                cv2.fillPoly(score_map    , np.round(stcl_quads           ).astype(np.int32), 1.0)
                cv2.fillPoly(score_map_big, np.round(stcl_quads / ds_ratio).astype(np.int32), 1.0)

                for idx, quad in enumerate(stcl_quads):
                    quad_mask = np.zeros((h, w), dtype=np.float32)
                    quad_mask = cv2.fillPoly(quad_mask, np.round(quad[np.newaxis, :, :]).astype(np.int32), 1.0)
                    tbo_map = self.gen_quad_tbo(poly_quads[quad_index[idx]], quad_mask, tbo_map)

                # score label map and score_label_map_text_label_list for refine
                if label_idx == 0:
                    text_pos_list_ = [[len(self.vocabulary)], ]
                    score_label_map_text_label_list.append(text_pos_list_)

                label_idx += 1
                cv2.fillPoly(score_label_map, np.round(poly_quads).astype(np.int32), label_idx)
                score_label_map_text_label_list.append(text_label_index_list)

                # direction info, fix-me
                n_char = len(text_label_index_list)
                direction_map = self.generate_direction_map(poly_quads, n_char, direction_map)

                # pos info
                average_shrink_height = self.calculate_average_height(stcl_quads)
                pos_l, pos_m = self.fit_and_gather_tcl_points_v2(min_area_quad, poly, max_h=h, max_w=w,
                                                                 fixed_point_num=64, image_id=self.image_id,
                                                                 reference_height=average_shrink_height)

                label_l = text_label_index_list
                # if len(text_label_index_list) < 2:
                #     continue

                pos_list.append(pos_l)
                pos_mask.append(pos_m)
                label_list.append(label_l)

        # use big score_map for smooth tcl lines
        score_map_big_resized = cv2.resize(score_map_big, dsize=None, fx=ds_ratio, fy=ds_ratio)
        score_map             =   np.array(score_map_big_resized > EPS, dtype='float32')

        return score_map, score_label_map, tbo_map, direction_map, training_mask, \
               pos_list, pos_mask, label_list, score_label_map_text_label_list

    def adjust_point(self, poly):
        """
        adjust point order.
        """
        point_num = poly.shape[0]
        if point_num == 4:
            len_1 = np.linalg.norm(poly[0] - poly[1])
            len_2 = np.linalg.norm(poly[1] - poly[2])
            len_3 = np.linalg.norm(poly[2] - poly[3])
            len_4 = np.linalg.norm(poly[3] - poly[0])

            if (len_1 + len_3) * 1.5 < (len_2 + len_4):
                poly = poly[[1, 2, 3, 0], :]

        elif point_num > 4:
            vector_1 = poly[0] - poly[1]
            vector_2 = poly[1] - poly[2]
            cos_theta = np.dot(vector_1, vector_2) / (
                np.linalg.norm(vector_1) * np.linalg.norm(vector_2) + EPS)
            theta = np.arccos(np.round(cos_theta, decimals=4))

            if abs(theta) > (70 / 180 * math.pi):
                index = list(range(1, point_num)) + [0]
                poly = poly[np.array(index), :]
        return poly

    def gen_min_area_quad_from_poly(self, poly):
        """
        Generate min area quad from poly.
        """
        point_num = poly.shape[0]
        min_area_quad = np.zeros((4, 2), dtype=np.float32)
        if point_num == 4:
            min_area_quad = poly
            center_point = np.sum(poly, axis=0) / 4
        else:
            rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
            center_point = rect[0]
            box = np.array(cv2.boxPoints(rect))

            first_point_idx = 0
            min_dist = 1e4
            for i in range(4):
                dist = np.linalg.norm(box[(i + 0) % 4] - poly[0                 ]) + \
                       np.linalg.norm(box[(i + 1) % 4] - poly[point_num // 2 - 1]) + \
                       np.linalg.norm(box[(i + 2) % 4] - poly[point_num // 2    ]) + \
                       np.linalg.norm(box[(i + 3) % 4] - poly[-1                ])
                if dist < min_dist:
                    min_dist = dist
                    first_point_idx = i

            for i in range(4):
                min_area_quad[i] = box[(first_point_idx + i) % 4]

        return min_area_quad, center_point

    def shrink_quad_along_width(self, quad, begin_width_ratio=0., end_width_ratio=1.):
        """
        Generate shrink_quad_along_width.
        """
        ratio_pair = np.array([[begin_width_ratio], [end_width_ratio]], dtype=np.float32)
        p0_1 = quad[0] + (quad[1] - quad[0]) * ratio_pair
        p3_2 = quad[3] + (quad[2] - quad[3]) * ratio_pair
        return np.array([p0_1[0], p0_1[1], p3_2[1], p3_2[0]])

    def shrink_poly_along_width(self, quads, shrink_ratio_of_width, expand_height_ratio=1.0):
        """
        shrink poly with given length.
        """
        upper_edge_list = []

        def get_cut_info(edge_len_list, cut_len):
            for idx, edge_len in enumerate(edge_len_list):
                cut_len -= edge_len
                if cut_len <= EPS:
                    ratio = (cut_len + edge_len_list[idx]) / edge_len_list[idx]
                    return idx, ratio

        for quad in quads:
            upper_edge_len = np.linalg.norm(quad[0] - quad[1])
            upper_edge_list.append(upper_edge_len)

        # length of left edge and right edge.
        left_length  = np.linalg.norm(quads[ 0][0] - quads[ 0][3]) * expand_height_ratio
        right_length = np.linalg.norm(quads[-1][1] - quads[-1][2]) * expand_height_ratio

        shrink_length = min(left_length, right_length, sum(upper_edge_list)) * shrink_ratio_of_width

        # shrinking length
        upper_len_left = shrink_length
        upper_len_right = sum(upper_edge_list) - shrink_length

        left_idx ,  left_ratio = get_cut_info(upper_edge_list, upper_len_left)
        right_idx, right_ratio = get_cut_info(upper_edge_list, upper_len_right)
        left_quad  = self.shrink_quad_along_width(quads[ left_idx], begin_width_ratio=left_ratio, end_width_ratio=          1)
        right_quad = self.shrink_quad_along_width(quads[right_idx], begin_width_ratio=         0, end_width_ratio=right_ratio)

        out_quad_list = []
        if left_idx == right_idx:
            out_quad_list.append([left_quad[0], right_quad[1], right_quad[2], left_quad[3]])
        else:
            out_quad_list.append(left_quad)
            for idx in range(left_idx+1, right_idx):
                out_quad_list.append(quads[idx])
            out_quad_list.append(right_quad)

        return np.array(out_quad_list), list(range(left_idx, right_idx+1))

    def prepare_text_label(self, label_str):
        """
        Prepare text lablel by default vocabulary.
        """
        if len(self.vocabulary) == 36:
            return label_str.lower()
        else:
            return label_str

    def vector_angle(self, A, B):
        """
        Calculate the angle between vector AB and x-axis positive direction.
        """
        AB = np.array([B[1] - A[1], B[0] - A[0]])
        return np.arctan2(*AB)

    def theta_line_cross_point(self, theta, point):
        """
        Calculate the line through given point and angle in ax + by + c =0 form.
        """
        x, y = point
        cos = np.cos(theta)
        sin = np.sin(theta)
        return [sin, -cos, cos * y - sin * x]

    def line_cross_two_point(self, A, B):
        """
        Calculate the line through given point A and B in ax + by + c =0 form.
        """
        angle = self.vector_angle(A, B)
        return self.theta_line_cross_point(angle, A)

    def average_angle(self, poly):
        """
        Calculate the average angle between left and right edge in given poly.
        """
        p0, p1, p2, p3 = poly
        angle30 = self.vector_angle(p3, p0)
        angle21 = self.vector_angle(p2, p1)
        return (angle30 + angle21) / 2

    def line_cross_point(self, line1, line2):
        """
        line1 and line2 in  0=ax+by+c form, compute the cross point of line1 and line2
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        d = a1 * b2 - a2 * b1

        if d == 0:
            print('Cross point does not exist')
            return np.array([0, 0], dtype=np.float32)
        else:
            x = (b1 * c2 - b2 * c1) / d
            y = (a2 * c1 - a1 * c2) / d

        return np.array([x, y], dtype=np.float32)

    def quad2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point. (4, 2)
        """
        ratio_pair = np.array([[0.5 - ratio / 2],
                               [0.5 + ratio / 2],], dtype=np.float32)
        p0_3 = poly[0] + (poly[3] - poly[0]) * ratio_pair
        p1_2 = poly[1] + (poly[2] - poly[1]) * ratio_pair
        return np.array([p0_3[0], p1_2[0], p1_2[1], p0_3[1]])

    def poly2tcl(self, poly, ratio):
        """
        Generate center line by poly clock-wise point.
        """
        ratio_pair = np.array([[0.5 - ratio / 2], 
                               [0.5 + ratio / 2]], dtype=np.float32)
        tcl_poly = np.zeros_like(poly)
        point_num = poly.shape[0]

        for idx in range(point_num // 2):
            point_pair = poly[idx] + (poly[point_num - 1 - idx] - poly[idx]) * ratio_pair
            tcl_poly[                idx] = point_pair[0]
            tcl_poly[point_num - 1 - idx] = point_pair[1]
        return tcl_poly

    def gen_quad_tbo(self, quad, tcl_mask, tbo_map):
        """
        Generate tbo_map for give quad.
        """
        # upper and lower line function: ax + by + c = 0;
        upper_line = self.line_cross_two_point(quad[0], quad[1])
        lower_line = self.line_cross_two_point(quad[3], quad[2])

        quad_h = 0.5 * (np.linalg.norm(quad[0] - quad[3]) +
                        np.linalg.norm(quad[1] - quad[2]))
        quad_w = 0.5 * (np.linalg.norm(quad[0] - quad[1]) +
                        np.linalg.norm(quad[2] - quad[3]))

        # average angle of left and right line.
        angle = self.average_angle(quad)

        xy_in_poly = np.argwhere(tcl_mask == 1)
        for y, x in xy_in_poly:
            point = (x, y)
            line        = self.theta_line_cross_point(angle, point)
            cross_point_upper = self.line_cross_point(upper_line, line)
            cross_point_lower = self.line_cross_point(lower_line, line)

            # FIX, offset reverse
            upper_offset_x, upper_offset_y = cross_point_upper - point
            lower_offset_x, lower_offset_y = cross_point_lower - point
            tbo_map[y, x, 0] = upper_offset_y
            tbo_map[y, x, 1] = upper_offset_x
            tbo_map[y, x, 2] = lower_offset_y
            tbo_map[y, x, 3] = lower_offset_x
            tbo_map[y, x, 4] = 1.0 / max(min(quad_h, quad_w), 1.0) * 2
        return tbo_map

    def poly2quads(self, poly):
        """
        Split poly into quads.
        """
        quad_list = []
        point_num = poly.shape[0]

        # point pair
        point_pair_list = []
        for idx in range(point_num // 2):
            point_pair = [poly[idx], poly[point_num - 1 - idx]]
            point_pair_list.append(point_pair)

        quad_num = point_num // 2 - 1
        for idx in range(quad_num):
            # reshape and adjust to clock-wise
            quad_list.append((np.array(point_pair_list)[[idx, idx + 1]]).reshape(4, 2)[[0, 2, 3, 1]])

        return np.array(quad_list)

    def rotate_im_poly(self, im, text_polys):
        """
        rotate image with 90 / 180 / 270 degre
        """
        im_w, im_h = im.shape[1], im.shape[0]
        dst_im = im.copy()
        dst_polys = []
        rand_degree_ratio = np.random.rand()
        rand_degree_cnt = 1
        if rand_degree_ratio > 0.5:
            rand_degree_cnt = 3
        for i in range(rand_degree_cnt):
            dst_im = np.rot90(dst_im)
        rot_degree = -90 * rand_degree_cnt
        rot_angle = rot_degree * math.pi / 180.0
        n_poly = text_polys.shape[0]
        
        cx, ncx = 0.5 * im_w, 0.5 * dst_im.shape[1]
        cy, ncy = 0.5 * im_h, 0.5 * dst_im.shape[0]

        for i in range(n_poly):
            wordBB = text_polys[i]
            poly = []
            for j in range(4):  # 16->4
                sx, sy = wordBB[j][0], wordBB[j][1]
                dx = math.cos(rot_angle) * (sx - cx) - math.sin(rot_angle) * (sy - cy) + ncx
                dy = math.sin(rot_angle) * (sx - cx) + math.cos(rot_angle) * (sy - cy) + ncy
                poly.append([dx, dy])
            dst_polys.append(poly)
        return dst_im, np.array(dst_polys, dtype=np.float32)
