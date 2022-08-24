import json
import numpy as np
from copy import deepcopy
from shapely.geometry import LineString, Point, Polygon

from src.data.label import RecLabelEncoder
from src.models.heads.recognition.SAR.label import SARLabelEncoder
from src.models.heads.recognition.CTC.label import CTCLabelEncoder


class MultiLabelEncoder(RecLabelEncoder):

    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs):
        super(MultiLabelEncoder, self).__init__(max_text_length, character_dict_path, use_space_char)

        self.ctc_encode = CTCLabelEncoder(max_text_length, character_dict_path, use_space_char, **kwargs)
        self.sar_encode = SARLabelEncoder(max_text_length, character_dict_path, use_space_char, **kwargs)

    def __call__(self, data):

        data_ctc = deepcopy(data)
        data_sar = deepcopy(data)
        data_out = dict()
        data_out['img_path'] = data.get('img_path', None)
        data_out['image'] = data['image']
        ctc = self.ctc_encode.__call__(data_ctc)
        sar = self.sar_encode.__call__(data_sar)
        if ctc is None or sar is None:
            return None

        data_out['label_sar'] = sar['label']
        data_out['label_ctc'] = ctc['label']
        data_out['length'   ] = ctc['length']
        return data_out


class OCRLabelEncoder(RecLabelEncoder):

    def __init__(self, encode_text, max_text_len=None, character_list=None, use_space_char=False, **kwargs):
        if encode_text:
            super(OCRLabelEncoder, self).__init__(max_text_len, character_list, use_space_char)
        self.encode_text = encode_text
        self.label_attr = kwargs.get('label_attr', 'label')
        self.text_attr = kwargs.get('text_attr', 'text')
        self.bbox_attr = kwargs.get('bbox_attr', 'polygons')

    def __call__(self, data):
        if self.encode_text:
            vocab_size = len(self.vocab)

        # Process labels for detection
        labels = data[self.label_attr]
        if isinstance(labels, str):
            labels = json.loads(labels)
        if len(labels) == 0:
            return None

        boxes, texts, ig_tags = [], [], []
        for box, txt in zip(labels[self.bbox_attr], 
                            labels[self.text_attr]):
            boxes.append(box)
            texts.append(txt)
            ig_tags.append(True if txt in ['*', '###'] else False)   # to ignore 

        boxes = np.array(boxes, dtype=np.float32)
        texts = np.array(texts)
        ig_tags = np.array(ig_tags, dtype=np.bool8)

        data['polys'] = boxes
        data['texts'] = texts
        data['ignore_tags'] = ig_tags

        # Process labels for recognition
        if self.encode_text:
            idx_texts = []
            for text in texts:
                text = self.encode(text.lower())
                if text is None:
                    return None
                text = text + [vocab_size] * (self.max_text_len - len(text))  # use 36 to pad
                idx_texts.append(text)
            data['texts'] = np.array(idx_texts)

        return data


class KIELabelEncoder(object):

    def __init__(self, character_dict_path, norm=10, directed=False, **kwargs):
        super(KIELabelEncoder, self).__init__()
        self.dict = dict({'': 0})
        with open(character_dict_path, 'r', encoding='utf-8') as fr:
            idx = 1
            for line in fr:
                char = line.strip()
                self.dict[char] = idx
                idx += 1
        self.norm = norm
        self.directed = directed

    def compute_relation(self, boxes):
        """
        Compute relation between every two boxes.
        """
        x1s, y1s = boxes[:, 0:1], boxes[:, 1:2]
        x2s, y2s = boxes[:, 4:5], boxes[:, 5:6]
        ws =            x2s - x1s + 1
        hs = np.maximum(y2s - y1s + 1, 1)
        dxs = (x1s[:, 0][None] - x1s) / self.norm
        dys = (y1s[:, 0][None] - y1s) / self.norm
        xhhs =  hs[:, 0][None] / hs
        xwhs =  ws[:, 0][None] / hs
        whs = ws / hs + np.zeros_like(xhhs)
        relations = np.stack([dxs, dys, whs, xhhs, xwhs], -1)
        bboxes = np.concatenate([x1s, y1s, x2s, y2s], -1).astype(np.float32)
        return relations, bboxes

    def pad_text_indices(self, text_ids):
        """
        Pad text index to same length.
        """
        max_len = 300
        recoder_len = max([len(text_id) for text_id in text_ids])
        padded_text_ids = -np.ones((len(text_ids), max_len), np.int32)
        for idx, text_id in enumerate(text_ids):
            padded_text_ids[idx, :len(text_id)] = np.array(text_id)
        return padded_text_ids, recoder_len

    def list_to_numpy(self, ann_infos):
        """
        Convert bboxes, relations, texts and labels to ndarray.
        """
        boxes, text_inds = ann_infos['points'], ann_infos['text_inds']
        boxes = np.array(boxes, np.int32)
        relations, bboxes = self.compute_relation(boxes)

        labels = ann_infos.get('labels', None)
        if labels is not None:
            labels = np.array(labels, np.int32)
            edges = ann_infos.get('edges', None)
            if edges is not None:
                labels = labels[:, None]
                edges = np.array(edges)
                edges = (edges[:, None] == edges[None, :]).astype(np.int32)
                if self.directed:
                    edges = (edges & labels == 1).astype(np.int32)
                np.fill_diagonal(edges, -1)
                labels = np.concatenate([labels, edges], -1)
        padded_text_inds, recoder_len = self.pad_text_indices(text_inds)
        max_num = 300
        temp_bboxes = np.zeros([max_num, 4])
        h, _ = bboxes.shape
        temp_bboxes[:h, :] = bboxes

        temp_relations = np.zeros([max_num, max_num, 5])
        temp_relations[:h, :h, :] = relations

        temp_padded_text_inds = np.zeros([max_num, max_num])
        temp_padded_text_inds[:h, :] = padded_text_inds

        temp_labels = np.zeros([max_num, max_num])
        temp_labels[:h, :h + 1] = labels

        tag = np.array([h, recoder_len])
        return dict(
            image=ann_infos['image'],
            points=temp_bboxes,
            relations=temp_relations,
            texts=temp_padded_text_inds,
            labels=temp_labels,
            tag=tag)

    def convert_canonical(self, points_x, points_y):

        assert len(points_x) == 4
        assert len(points_y) == 4

        points = [Point(points_x[i], points_y[i]) for i in range(4)]

        polygon = Polygon([(p.x, p.y) for p in points])
        min_x, min_y, _, _ = polygon.bounds
        points_to_lefttop = [
            LineString([points[i], Point(min_x, min_y)]) for i in range(4)
        ]
        distances = np.array([line.length for line in points_to_lefttop])
        sort_dist_idx = np.argsort(distances)
        lefttop_idx = sort_dist_idx[0]

        if lefttop_idx == 0:
            point_orders = [0, 1, 2, 3]
        elif lefttop_idx == 1:
            point_orders = [1, 2, 3, 0]
        elif lefttop_idx == 2:
            point_orders = [2, 3, 0, 1]
        else:
            point_orders = [3, 0, 1, 2]

        sorted_points_x = [points_x[i] for i in point_orders]
        sorted_points_y = [points_y[j] for j in point_orders]

        return sorted_points_x, sorted_points_y

    def sort_vertex(self, points_x, points_y):

        assert len(points_x) == 4
        assert len(points_y) == 4

        x = np.array(points_x)
        y = np.array(points_y)
        center_x = np.sum(x) * 0.25
        center_y = np.sum(y) * 0.25

        x_arr = np.array(x - center_x)
        y_arr = np.array(y - center_y)

        angle = np.arctan2(y_arr, x_arr) * 180.0 / np.pi
        sort_idx = np.argsort(angle)

        sorted_points_x, sorted_points_y = [], []
        for i in range(4):
            sorted_points_x.append(points_x[sort_idx[i]])
            sorted_points_y.append(points_y[sort_idx[i]])

        return self.convert_canonical(sorted_points_x, sorted_points_y)

    def __call__(self, data):
        label = data['label']
        annotations = json.loads(label)
        boxes, texts, text_inds, labels, edges = [], [], [], [], []
        for ann in annotations:
            box = ann['points']
            x_list = [box[i][0] for i in range(4)]
            y_list = [box[i][1] for i in range(4)]
            sorted_x_list, sorted_y_list = self.sort_vertex(x_list, y_list)
            sorted_box = []
            for x, y in zip(sorted_x_list, sorted_y_list):
                sorted_box.append(x)
                sorted_box.append(y)
            boxes.append(sorted_box)
            text       = ann['transcription']
            texts.append(ann['transcription'])
            text_ind = [self.dict[c] for c in text if c in self.dict]
            text_inds.append(text_ind)
            if 'label' in ann.keys():
                labels.append(ann['label'])
            elif 'key_cls' in ann.keys():
                labels.append(ann['key_cls'])
            else:
                raise ValueError("Cannot found 'key_cls' in ann.keys(), please check your training annotation.")

            edges.append(ann.get('edge', 0))
        
        ann_infos = dict(image=data['image'],
                        points=boxes,
                         texts=texts,
                     text_inds=text_inds,
                         edges=edges,
                        labels=labels)

        return self.list_to_numpy(ann_infos)





