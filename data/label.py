import json
import numpy as np
import pandas as pd


class ClsLabelEncoder(object):
    """
    Multi-class Label Encoder
    """
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, data):
        label = data['label']
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data['label'] = label
        return data


class DetLabelEncoder(object):
    """
    Detection Label Encoder
    """
    def __init__(self, label_attr: str = 'label', 
                        bbox_attr: str = 'polygons', 
                        text_attr: str = 'text', **kwargs):
        self.text_attr = text_attr
        self.bbox_attr = bbox_attr
        self.label_attr = label_attr

    def __call__(self, data):
        labels = data[self.label_attr]
        if isinstance(labels, str):
            labels = json.loads(labels)
        if len(labels) == 0:
            return None

        boxes, texts, ig_tags = [], [], []
        for label in labels:
            box = label[self.bbox_attr]
            txt = label[self.text_attr]
            boxes.append(box)
            texts.append(txt)
            ig_tags.append(True if txt in ['*', '###'] else False)   # to ignore 

        boxes = self.expand_points_num(boxes)
        boxes   = np.array(boxes, dtype=np.float32)
        ig_tags = np.array(ig_tags, dtype=np.bool8)

        data['polys'] = boxes
        data['texts'] = texts
        data['ignore_tags'] = ig_tags
        return data

    def expand_points_num(self, boxes):
        """
        Ensure all bboxes having the same number of coordinates
        """
        max_points = 0
        for box in boxes:
            if len(box) > max_points:
                max_points = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect


class RecLabelEncoder(object):
    """ 
    Convert between text-label and text-index 
    """
    def __init__(self, max_text_len, character_list: list = None, use_space_char: bool = False):

        self.max_text_len = max_text_len
        self.beg_token = "sos"
        self.end_token = "eos"
        self.lowercase = False

        if character_list is None:
            print("Default vocabulary includes 10 numbers and 24 English lower-cased letters")
            character_list = list("0123456789abcdefghijklmnopqrstuvwxyz")
            self.lowercase = True

        if use_space_char:
            space_char = ' '
            if space_char not in character_list:
                character_list.append(space_char)

        self.vocab = {}
        for ci, char in enumerate(character_list):
            self.vocab[char] = ci

    def add_special_char(self, new_characters):
        if isinstance(new_characters, (str,)):
            new_characters = list(new_characters)
        vocab_size = len(self.vocab)
        for ci, char in enumerate(new_characters):
            self.vocab[char] = vocab_size + ci

    def encode(self, text):
        """
        convert text-label into text-index.
        
        input:
        ------
        text: text labels of each image. [batch_size]
        
        output:
        -------
        text: concatenated text index for CTCLoss.
                [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
        length: length of each text. [batch_size]
        """
        if len(text) == 0:
            return None
        if len(text) > self.max_text_len:
            text = text[:self.max_text_len]
        if self.lowercase:
            text = text.lower()

        text_indices = []
        for char in text:
            if char not in self.vocab.keys():
                print(f"Character {char} is not found in vocabulary!")
                continue
            text_indices.append(self.vocab[char])
        if len(text_indices) > 0:
            return text_indices
        return None



