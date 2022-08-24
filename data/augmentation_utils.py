import cv2
import numpy as np

from shapely.geometry import Polygon


#####################################################################
#                        Image Augmentation                         #
#####################################################################

def get_union(pD, pG):
    return Polygon(pD).union(Polygon(pG)).area


def get_intersection_over_union(pD, pG):
    return get_intersection(pD, pG) / get_union(pD, pG)


def get_intersection(pD, pG):
    return Polygon(pD).intersection(Polygon(pG)).area


def rotate_bbox(img, text_polys, angle, scale=1):
    """
    from https://github.com/WenmuZhou/DBNet.pytorch/blob/master/data_loader/modules/augment.py
    Args:
        img: np.ndarray
        text_polys: np.ndarray N*4*2
        angle: int
        scale: int
    Returns:
    """
    w = img.shape[1]
    h = img.shape[0]

    rangle = np.deg2rad(angle)
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]

    # rotate box
    rot_text_polys = list()
    for bbox in text_polys:
        point1 = np.dot(rot_mat, np.array([bbox[0, 0], bbox[0, 1], 1]))
        point2 = np.dot(rot_mat, np.array([bbox[1, 0], bbox[1, 1], 1]))
        point3 = np.dot(rot_mat, np.array([bbox[2, 0], bbox[2, 1], 1]))
        point4 = np.dot(rot_mat, np.array([bbox[3, 0], bbox[3, 1], 1]))
        rot_text_polys.append([point1, point2, point3, point4])
    return np.array(rot_text_polys, dtype=np.float32)


def is_poly_inside_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
        return False
    if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
        return False
    return True


def is_poly_outside_rect(poly, x, y, w, h):
    poly = np.array(poly)
    if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
        return True
    if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
        return True
    return False


def split_regions(axis):
    regions = []
    min_axis = 0
    for i in range(1, axis.shape[0]):
        if axis[i] != axis[i - 1] + 1:
            region = axis[min_axis:i]
            min_axis = i
            regions.append(region)
    return regions


def random_select(axis, max_size):
    xx = np.random.choice(axis, size=2)
    xmin = np.min(xx)
    xmax = np.max(xx)
    xmin = np.clip(xmin, 0, max_size - 1)
    xmax = np.clip(xmax, 0, max_size - 1)
    return xmin, xmax


def region_wise_random_select(regions, max_size):
    selected_index = list(np.random.choice(len(regions), 2))
    selected_values = []
    for index in selected_index:
        axis = regions[index]
        xx = int(np.random.choice(axis, size=1))
        selected_values.append(xx)
    xmin = min(selected_values)
    xmax = max(selected_values)
    return xmin, xmax


def crop_area(im, text_polys, min_crop_side_ratio, max_tries):
    h, w, _ = im.shape
    h_array = np.zeros(h, dtype=np.int32)
    w_array = np.zeros(w, dtype=np.int32)
    for points in text_polys:
        points = np.round(points, decimals=0).astype(np.int32)
        minx = np.min(points[:, 0])
        maxx = np.max(points[:, 0])
        w_array[minx:maxx] = 1
        miny = np.min(points[:, 1])
        maxy = np.max(points[:, 1])
        h_array[miny:maxy] = 1

    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]

    if len(h_axis) == 0 or len(w_axis) == 0:
        return 0, 0, w, h

    h_regions = split_regions(h_axis)
    w_regions = split_regions(w_axis)

    for i in range(max_tries):
        if len(w_regions) > 1:
            xmin, xmax = region_wise_random_select(w_regions, w)
        else:
            xmin, xmax = random_select(w_axis, w)
        if len(h_regions) > 1:
            ymin, ymax = region_wise_random_select(h_regions, h)
        else:
            ymin, ymax = random_select(h_axis, h)

        if xmax - xmin < min_crop_side_ratio * w or ymax - ymin < min_crop_side_ratio * h:
            # area too small
            continue
        num_poly_in_rect = 0
        for poly in text_polys:
            if not is_poly_outside_rect(poly, xmin, ymin, xmax - xmin,
                                        ymax - ymin):
                num_poly_in_rect += 1
                break

        if num_poly_in_rect > 0:
            return xmin, ymin, xmax - xmin, ymax - ymin

    return 0, 0, w, h


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]),
                             np.linalg.norm(points[2] - points[3]),))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]),
                              np.linalg.norm(points[1] - points[2]),))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                                  [img_crop_width, img_crop_height],
                                               [0, img_crop_height],])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height),
                                  borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img


#####################################################################
#                  Text-Image Augmentation [TIA]                    #
#####################################################################

class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1. / ( (i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0]) +
                                  (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1]))

                    sw +=       w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] =  np.sum(pt_i * cur_pt  ) * self.src_pts[k][0] - \
                                     np.sum(pt_j * cur_pt  ) * self.src_pts[k][1]
                        tmp_pt[1] = -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + \
                                     np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar

                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(di / h, dj / w, self.rdx[ i, j], self.rdx[ i, nj],
                                                                 self.rdx[ni, j], self.rdx[ni, nj],)
                delta_y = self.__bilinear_interp(di / h, dj / w, self.rdy[ i, j], self.rdy[ i, nj],
                                                                 self.rdy[ni, j], self.rdy[ni, nj],)
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h, j:j + w] = self.__bilinear_interp(x, y, self.src[nyi , nxi], self.src[nyi , nxi1],
                                                                     self.src[nyi1, nxi], self.src[nyi1, nxi1],)

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return 


def tia_distort(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([        np.random.randint(thresh),         np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh),         np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append([        np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                                        np.random.randint(thresh) - half_thresh,])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                                img_h + np.random.randint(thresh) - half_thresh,])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([    0,         np.random.randint(thresh)])
    dst_pts.append([img_w,         np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([    0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst
