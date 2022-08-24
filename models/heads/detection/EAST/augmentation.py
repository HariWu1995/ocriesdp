import cv2
import numpy as np

from src.data.augmentation_utils import crop_area, is_poly_outside_rect


class EastRandomCropData(object):
    
    def __init__(self, size=(640, 640), max_tries=10, min_crop_side_ratio=0.1, keep_ratio=True, **kwargs):
        self.size = size
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.keep_ratio = keep_ratio

    def __call__(self, data):
        img = data['image']
        text_polys = data['polys']
        ignore_tags = data['ignore_tags']
        texts = data['texts']
        all_care_polys = [
            text_polys[i] for i, tag in enumerate(ignore_tags) if not tag
        ]

        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = crop_area(img, all_care_polys, self.min_crop_side_ratio, self.max_tries)
        
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            padimg = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
            padimg[:h, :w] = cv2.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(
                img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                tuple(self.size))
        # crop 文本框
        text_polys_crop = []
        ignore_tags_crop = []
        texts_crop = []
        for poly, text, tag in zip(text_polys, texts, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)
                texts_crop.append(text)
        data['image'] = img
        data['polys'] = np.array(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        data['texts'] = texts_crop
        return data

