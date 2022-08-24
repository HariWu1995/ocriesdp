from copy import deepcopy
import json
import numpy as np

from transformers import LayoutXLMTokenizer, LayoutLMTokenizer, LayoutLMv2Tokenizer


def load_vqa_bio_label_maps(label_map_path):
    with open(label_map_path, "r", encoding='utf-8') as fin:
        lines = fin.readlines()
    lines = [line.strip() for line in lines]
    if "O" not in lines:
        lines.insert(0, "O")
    labels = []
    for line in lines:
        if line == "O":
            labels.append("O")
        else:
            labels.append("B-" + line)
            labels.append("I-" + line)
    label2id_map = {label: idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


class VQATokenLabelEncoder(object):
    """
    Label encode for NLP VQA methods
    """
    def __init__(self, class_path, contains_re=False, add_special_ids=False,
                    algorithm='LayoutXLM', infer_mode=False, ocr_engine=None, **kwargs):
        super(VQATokenLabelEncoder, self).__init__()
        tokenizer_dict = {
            'LayoutXLM': {
                'class': LayoutXLMTokenizer,
                'pretrained_model': 'layoutxlm-base-uncased'
            },
            'LayoutLM': {
                'class': LayoutLMTokenizer,
                'pretrained_model': 'layoutlm-base-uncased'
            },
            'LayoutLMv2': {
                'class': LayoutLMv2Tokenizer,
                'pretrained_model': 'layoutlmv2-base-uncased'
            }
        }
        self.contains_re = contains_re
        tokenizer_config = tokenizer_dict[algorithm]
        self.tokenizer = tokenizer_config['class'].from_pretrained(
            tokenizer_config['pretrained_model'])
        self.label2id_map, id2label_map = load_vqa_bio_label_maps(class_path)
        self.add_special_ids = add_special_ids
        self.infer_mode = infer_mode
        self.ocr_engine = ocr_engine

    def __call__(self, data):
        # load bbox and label info
        ocr_info = self._load_ocr_info(data)

        height, width, _ = data['image'].shape

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        segment_offset_id = []
        gt_label_list = []
        entities = []

        # for re
        train_re = self.contains_re and not self.infer_mode
        if train_re:
            relations = []
            id2label = {}
            entity_id_to_index_map = {}
            empty_entity = set()

        data['ocr_info'] = deepcopy(ocr_info)

        for info in ocr_info:
            if train_re:
                # for re
                if len(info["text"]) == 0:
                    empty_entity.add(info["id"])
                    continue
                id2label[info["id"]] = info["label"]
                relations.extend([tuple(sorted(l)) for l in info["linking"]])

            # smooth_box
            bbox = self._smooth_box(info["bbox"], height, width)

            text = info["text"]
            encode_res = self.tokenizer.encode(
                text, pad_to_max_seq_len=False, return_attention_mask=True)

            if not self.add_special_ids:
                # TODO: use tok.all_special_ids to remove
                encode_res[     "input_ids"] = encode_res[     "input_ids"][1:-1]
                encode_res["token_type_ids"] = encode_res["token_type_ids"][1:-1]
                encode_res["attention_mask"] = encode_res["attention_mask"][1:-1]

            # parse label
            if not self.infer_mode:
                label = info['label']
                gt_label = self._parse_label(label, encode_res)

            # construct entities for re
            if train_re:
                if gt_label[0] != self.label2id_map["O"]:
                    entity_id_to_index_map[info["id"]] = len(entities)
                    label = label.upper()
                    entities.append({
                        "start": len(input_ids_list),
                          "end": len(input_ids_list) + len(encode_res["input_ids"]),
                        "label": label.upper(),
                    })
            else:
                entities.append({
                    "start": len(input_ids_list),
                      "end": len(input_ids_list) + len(encode_res["input_ids"]),
                    "label": 'O',
                })
            input_ids_list.extend(encode_res["input_ids"])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend([bbox] * len(encode_res["input_ids"]))
            words_list.append(text)
            segment_offset_id.append(len(input_ids_list))
            if not self.infer_mode:
                gt_label_list.extend(gt_label)

        data[     'input_ids'] = input_ids_list
        data['token_type_ids'] = token_type_ids_list
        data['bbox'] = bbox_list
        data['attention_mask'] = [1] * len(input_ids_list)
        data['labels'] = gt_label_list
        data['segment_offset_id'] = segment_offset_id
        data['tokenizer_params'] = dict(
                 padding_side=self.tokenizer.padding_side,
            pad_token_type_id=self.tokenizer.pad_token_type_id,
            pad_token_id     =self.tokenizer.pad_token_id
        )
        data['entities'] = entities

        if train_re:
            data['relations'] = relations
            data['id2label'] = id2label
            data['empty_entity'] = empty_entity
            data['entity_id_to_index_map'] = entity_id_to_index_map
        return data

    def _load_ocr_info(self, data):

        def trans_poly_to_bbox(poly):
            x1 = np.min([p[0] for p in poly])
            x2 = np.max([p[0] for p in poly])
            y1 = np.min([p[1] for p in poly])
            y2 = np.max([p[1] for p in poly])
            return [x1, y1, x2, y2]

        if self.infer_mode:
            ocr_result = self.ocr_engine.ocr(data['image'], cls=False)
            ocr_info = []
            for res in ocr_result:
                ocr_info.append({
                    "text": res[1][0],
                    "bbox": trans_poly_to_bbox(res[0]),
                    "poly": res[0],
                })
            return ocr_info
        else:
            info = data['label']
            # read text info
            info_dict = json.loads(info)
            return info_dict["ocr_info"]

    def _smooth_box(self, bbox, height, width):
        bbox[0] = int(bbox[0] * 1000.0 / width)
        bbox[2] = int(bbox[2] * 1000.0 / width)
        bbox[1] = int(bbox[1] * 1000.0 / height)
        bbox[3] = int(bbox[3] * 1000.0 / height)
        return bbox

    def _parse_label(self, label, encode_res):
        gt_label = []
        if label.lower() == "other":
            gt_label.extend([0] * len(encode_res["input_ids"]))
        else:
            gt_label.append( self.label2id_map[("b-" + label).upper()])
            gt_label.extend([self.label2id_map[("i-" + label).upper()]] *
                            (len(encode_res["input_ids"]) - 1))
        return gt_label
