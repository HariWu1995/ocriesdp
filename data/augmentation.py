import copy
import six

import cv2
import random as rd
import numpy as np

from PIL import Image, ImageOps, ImageEnhance
from shapely.geometry import Polygon

import imgaug
import imgaug.augmenters as iaa
import pyclipper

from src.data.augmentation_utils import get_rotate_crop_image, rotate_bbox, is_poly_outside_rect


class AugmenterBuilder(object):

    def __init__(self):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(*[self.formalize(a) for a in args[1:]])

        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{ k: self.formalize(v) for k, v in args['args'].items() })

        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def formalize(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class IaaAugment():

    def __init__(self, augmenter_args=None, **kwargs):
        if augmenter_args is None:
            augmenter_args = [{ 'type': 'Fliplr', 'args': { 'p': 0.5, }}, 
                              { 'type': 'Affine', 'args': { 'rotate': [-10, 10], }},
                              { 'type': 'Resize', 'args': { 'size': [0.5, 3], }}]
        self.augmenter = AugmenterBuilder().build(augmenter_args)

    def __call__(self, data):
        image = data['image']
        shape = image.shape

        if self.augmenter:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(image)
            data = self.may_augment_annotation(aug, data, shape)
        return data

    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data

        line_polys = []
        for poly in data['polys']:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_polys.append(new_poly)
        data['polys'] = np.array(line_polys)
        return data

    def may_augment_poly(self, aug, img_shape, poly):
        keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
        keypoints = aug.augment_keypoints(
            [imgaug.KeypointsOnImage(keypoints, shape=img_shape)])[0].keypoints
        poly = [(p.x, p.y) for p in keypoints]
        return poly


class CopyPaste(object):

    def __init__(self, objects_paste_ratio=0.2, limit_paste=True, **kwargs):
        self.ext_data_num = 1
        self.objects_paste_ratio = objects_paste_ratio
        self.limit_paste = limit_paste
        augmenter_args = [{'type': 'Resize', 'args': {'size': [0.5, 3]}}]
        self.aug = IaaAugment(augmenter_args)

    def __call__(self, data):
        point_num = data['polys'].shape[1]
        src_img   = data['image']
        src_polys = data['polys'].tolist()
        src_texts = data['texts']
        src_ignores = data['ignore_tags'].tolist()
        ext_data  =     data['ext_data'][0]
        ext_image = ext_data['image']
        ext_polys = ext_data['polys']
        ext_texts = ext_data['texts']
        ext_ignores = ext_data['ignore_tags']

        indexs = [i for i in range(len(ext_ignores)) if not ext_ignores[i]]
        select_num = max(1, min(int(self.objects_paste_ratio * len(ext_polys)), 30))

        rd.shuffle(indexs)
        select_idxs = indexs[:select_num]
        select_polys = ext_polys[select_idxs]
        select_ignores = ext_ignores[select_idxs]

        ext_image = cv2.cvtColor(ext_image, cv2.COLOR_BGR2RGB)
        src_img   = cv2.cvtColor(src_img  , cv2.COLOR_BGR2RGB)
        src_img = Image.fromarray(src_img).convert('RGBA')
        for idx, poly, tag in zip(select_idxs, select_polys, select_ignores):
            box_img = get_rotate_crop_image(ext_image, poly)

            src_img, box = self.paste_img(src_img, box_img, src_polys)
            if box is not None:
                box = box.tolist() 
                for _ in range(len(box), point_num):
                    box.append(box[-1])
                src_polys.append(box)
                src_texts.append(ext_texts[idx])
                src_ignores.append(tag)
        src_img = cv2.cvtColor(np.array(src_img), cv2.COLOR_RGB2BGR)
        h, w = src_img.shape[:2]
        src_polys = np.array(src_polys)
        src_polys[:, :, 0] = np.clip(src_polys[:, :, 0], 0, w)
        src_polys[:, :, 1] = np.clip(src_polys[:, :, 1], 0, h)
        data['image'] = src_img
        data['polys'] = src_polys
        data['texts'] = src_texts
        data['ignore_tags'] = np.array(src_ignores)
        return data

    def paste_img(self, src_img, box_img, src_polys):
        box_img_pil = Image.fromarray(box_img).convert('RGBA')
        src_w, src_h = src_img.size
        box_w, box_h = box_img_pil.size

        angle = np.random.randint(0, 360)
        box = np.array([[[0, 0], [box_w, 0], [box_w, box_h], [0, box_h]]])
        box = rotate_bbox(box_img, box, angle)[0]
        box_img_pil = box_img_pil.rotate(angle, expand=1)
        box_w, box_h = box_img_pil.width, box_img_pil.height
        if src_w - box_w < 0 or src_h - box_h < 0:
            return src_img, None

        paste_x, paste_y = self.select_coord(src_polys, box, src_w - box_w, src_h - box_h)
        if paste_x is None:
            return src_img, None
        box[:, 0] += paste_x
        box[:, 1] += paste_y
        r, g, b, A = box_img_pil.split()
        src_img.paste(box_img_pil, (paste_x, paste_y), mask=A)

        return src_img, box

    def select_coord(self, src_polys, box, endx, endy):
        if self.limit_paste:
            xmin, ymin = box[:, 0].min(), box[:, 1].min()
            xmax, ymax = box[:, 0].max(), box[:, 1].max()
            for _ in range(50):
                paste_x = rd.randint(0, endx)
                paste_y = rd.randint(0, endy)
                xmin1 = xmin + paste_x
                xmax1 = xmax + paste_x
                ymin1 = ymin + paste_y
                ymax1 = ymax + paste_y

                num_poly_in_rect = 0
                for poly in src_polys:
                    if not is_poly_outside_rect(poly, xmin1, ymin1, xmax1 - xmin1, ymax1 - ymin1):
                        num_poly_in_rect += 1
                        break
                if num_poly_in_rect == 0:
                    return paste_x, paste_y
            return None, None

        else:
            paste_x = rd.randint(0, endx)
            paste_y = rd.randint(0, endy)
            return paste_x, paste_y


class RandomCropImgMask(object):

    def __init__(self, size, main_key, crop_keys, p=3/8, **kwargs):
        self.size = size
        self.main_key = main_key
        self.crop_keys = crop_keys
        self.p = p

    def __call__(self, data):
        image = data['image']

        h, w = image.shape[0:2]
        th, tw = self.size
        if w == tw and h == th:
            return data

        mask = data[self.main_key]
        if np.max(mask) > 0 and rd.random() > self.p:

            # make sure to crop the text region
            tl = np.min(np.where(mask > 0), axis=1) - (th, tw)
            tl[tl < 0] = 0
            br = np.max(np.where(mask > 0), axis=1) - (th, tw)
            br[br < 0] = 0

            br[0] = min(br[0], h - th)
            br[1] = min(br[1], w - tw)

            i = rd.randint(tl[0], br[0]) if tl[0] < br[0] else 0
            j = rd.randint(tl[1], br[1]) if tl[1] < br[1] else 0

        else:
            i = rd.randint(0, h - th) if h - th > 0 else 0
            j = rd.randint(0, w - tw) if w - tw > 0 else 0

        # return i, j, th, tw
        for k in data:
            if k in self.crop_keys:
                if len(data[k].shape) == 3:
                    if np.argmin(data[k].shape) == 0:
                        img = data[k][:, i:i+th, j:j+tw]
                        if img.shape[1] != img.shape[2]:
                            a = 1
                    elif np.argmin(data[k].shape) == 2:
                        img = data[k][i:i+th, j:j+tw, :]
                        if img.shape[1] != img.shape[0]:
                            a = 1
                    else:
                        img = data[k]
                else:
                    img = data[k][i:i+th, j:j+tw]
                    if img.shape[0] != img.shape[1]:
                        a = 1
                data[k] = img
        return data


class RawRandAugment(object):

    def __init__(self, num_layers=2, magnitude=5, fillcolor=(128, 128, 128), **kwargs):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.max_level = 10

        abso_level = self.magnitude / self.max_level
        self.level_map = {
                "shearX" :         0.3 * abso_level,
                "shearY" :         0.3 * abso_level,
            "translateX" : 150.0 / 331 * abso_level,
            "translateY" : 150.0 / 331 * abso_level,
                "rotate" :          30 * abso_level,
             "posterize" :     int(4.0 * abso_level),
                 "color" :         0.9 * abso_level,
              "solarize" :       256.0 * abso_level,
              "contrast" :         0.9 * abso_level,
             "sharpness" :         0.9 * abso_level,
            "brightness" :         0.9 * abso_level,
          "autocontrast" : 0,
              "equalize" : 0,
                "invert" : 0
        }

        # from https://stackoverflow.com/questions/5252170/
        # specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128, ) * 4),
                                   rot).convert(img.mode)

        self.func = {
            "shearX": lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                          (1, magnitude * rd.choice([-1, 1]), 0, 0, 1, 0),
                                                          Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                          (1, 0, 0, magnitude * rd.choice([-1, 1]), 1, 0),
                                                          Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                          (1, 0, magnitude * img.size[0] * rd.choice([-1, 1]), 0, 1, 0),
                                                          fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(img.size, Image.AFFINE,
                                                          (1, 0, 0, 0, 1, magnitude * img.size[1] * rd.choice([-1, 1])),
                                                          fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * rd.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(1 + magnitude * rd.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(1 + magnitude * rd.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(1 + magnitude * rd.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def __call__(self, img):
        avaiable_op_names = list(self.level_map.keys())
        for layer_num in range(self.num_layers):
            op_name = np.random.choice(avaiable_op_names)
            img = self.func[op_name](img, self.level_map[op_name])
        return img


class RandAugment(RawRandAugment):
    """ 
    RandAugment wrapper to auto fit different img types 
    """
    def __init__(self, prob=0.5, *args, **kwargs):
        self.prob = prob
        if six.PY2:
            super(RandAugment, self).__init__(*args, **kwargs)
        else:
            super().__init__(*args, **kwargs)

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data
        img = data['image']
        if not isinstance(img, Image.Image):
            img = np.ascontiguousarray(img)
            img = Image.fromarray(img)

        if six.PY2:
            img = super(RandAugment, self).__call__(img)
        else:
            img = super().__call__(img)

        if isinstance(img, Image.Image):
            img = np.asarray(img)
        data['image'] = img
        return data


class MakeBorderMap(object):

    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7, **kwargs):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data):

        img         = data['image']
        text_polys  = data['polys']
        ignore_tags = data['ignore_tags']

        canvas = np.zeros(img.shape[:2], dtype=np.float32)
        mask   = np.zeros(img.shape[:2], dtype=np.float32)

        for i in range(len(text_polys)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(text_polys[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min

        data['threshold_map'] = canvas
        data['threshold_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim     == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        if polygon_shape.area <= 0:
            return
        distance = polygon_shape.area * (
            1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(np.linspace(0, width-1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(np.linspace(0, height-1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros((polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i+1) % polygon.shape[0]
            absolute_distance = self._distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid+1, 
               xmin_valid:xmax_valid+1] = np.fmax(1 - distance_map[ymin_valid-ymin : ymax_valid-ymax+height,
                                                                   xmin_valid-xmin : xmax_valid-xmax+width ,],
                                                            canvas[ymin_valid:ymax_valid+1, 
                                                                   xmin_valid:xmax_valid+1,])

    def _distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(     xs    - point_1[0]) + np.square(     ys    - point_1[1])
        square_distance_2 = np.square(     xs    - point_2[0]) + np.square(     ys    - point_2[1])
        square_distance   = np.square(point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / (2 * np.sqrt(square_distance_1 * square_distance_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 * square_sin / square_distance)

        result[cosin < 0] = np.sqrt(np.fmin(square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result, shrink_ratio):
        ex_point_1 = (
            int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + shrink_ratio))),
            int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + shrink_ratio))),
        )
        cv2.line(result, tuple(ex_point_1), tuple(point_1), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        
        ex_point_2 = (
            int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + shrink_ratio))),
            int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + shrink_ratio))),
        )
        cv2.line(result, tuple(ex_point_2), tuple(point_2), 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        
        return ex_point_1, ex_point_2


class MakeShrinkMap(object):
    '''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    def __init__(self, min_text_size=8, shrink_ratio=0.4, **kwargs):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio

    def __call__(self, data):
        image       = data['image']
        text_polys  = data['polys']
        ignore_tags = data['ignore_tags']

        h, w = image.shape[:2]
        text_polys, ignore_tags = self.validate_polygons(text_polys, ignore_tags, h, w)
        gt  = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(text_polys)):
            polygon = text_polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width  = max(polygon[:, 0]) - min(polygon[:, 0])
            
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True

            else:
                polygon_shape = Polygon(polygon)
                subject = [tuple(l) for l in polygon]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrinked = []

                # Increase the shrink ratio every time we get multiple polygon returned back
                possible_ratios = np.arange(self.shrink_ratio, 1, self.shrink_ratio)
                np.append(possible_ratios, 1)
                for ratio in possible_ratios:
                    # print(f"Change shrink ratio to {ratio}")
                    distance = polygon_shape.area * (1 - np.power(ratio, 2)) / polygon_shape.length
                    shrinked = padding.Execute(-distance)
                    if len(shrinked) == 1:
                        break

                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue

                for each_shirnk in shrinked:
                    shirnk = np.array(each_shirnk).reshape(-1, 2)
                    cv2.fillPoly(gt, [shirnk.astype(np.int32)], 1)

        data['shrink_map'] = gt
        data['shrink_mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        """
        compute polygon area
        """
        area = 0
        q = polygon[-1]
        for p in polygon:
            area += p[0] * q[1] - p[1] * q[0]
            q = p
        return area / 2.0


