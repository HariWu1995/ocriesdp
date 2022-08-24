import numpy as np


class TableAttentionLabelEncoder(object):
    """ 
    Convert between text-label and text-index 
    """
    def __init__(self, max_text_length, max_elem_length, max_cell_num, character_dict_path, span_weight=1.0, **kwargs):
        self.max_text_length = max_text_length
        self.max_elem_length = max_elem_length
        self.max_cell_num    = max_cell_num
        list_character, list_elem = self.load_char_elem_dict(character_dict_path)
        list_character = self.add_special_char(list_character)
        list_elem      = self.add_special_char(list_elem)
        self.dict_character = {}
        for i, char in enumerate(list_character):
            self.dict_character[char] = i
        self.dict_elem = {}
        for i, elem in enumerate(list_elem):
            self.dict_elem[elem] = i
        self.span_weight = span_weight

    def load_char_elem_dict(self, character_dict_path):
        list_character = []
        list_elem = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            substr = lines[0].decode('utf-8').strip("\r\n").split("\t")
            character_num = int(substr[0])
            elem_num = int(substr[1])
            for cno in range(1, 1 + character_num):
                character = lines[cno].decode('utf-8').strip("\r\n")
                list_character.append(character)
            for eno in range(1 + character_num, 1 + character_num + elem_num):
                elem = lines[eno].decode('utf-8').strip("\r\n")
                list_elem.append(elem)
        return list_character, list_elem

    def add_special_char(self, list_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        list_character = [self.beg_str] + list_character + [self.end_str]
        return list_character

    def get_span_idx_list(self):
        span_idx_list = []
        for elem in self.dict_elem:
            if 'span' in elem:
                span_idx_list.append(self.dict_elem[elem])
        return span_idx_list

    def __call__(self, data):
        cells = data['cells']
        structure = data['structure']['tokens']
        structure = self.encode(structure, 'elem')
        if structure is None:
            return None
        elem_num = len(structure)
        structure = [0] + structure + [len(self.dict_elem) - 1]
        structure = structure + [0] * (self.max_elem_length + 2 - len(structure))
        structure = np.array(structure)
        data['structure'] = structure
        elem_char_idx1 = self.dict_elem['<td>']
        elem_char_idx2 = self.dict_elem['<td']
        span_idx_list = self.get_span_idx_list()
        td_idx_list = np.logical_or(structure == elem_char_idx1,
                                    structure == elem_char_idx2)
        td_idx_list = np.where(td_idx_list)[0]

        structure_mask =  np.ones((self.max_elem_length + 2, 1), dtype=np.float32)
        bbox_list      = np.zeros((self.max_elem_length + 2, 4), dtype=np.float32)
        bbox_list_mask = np.zeros((self.max_elem_length + 2, 1), dtype=np.float32)
        
        img_height, img_width, img_ch = data['image'].shape
        if len(span_idx_list) > 0:
            span_weight = len(td_idx_list) * 1.0 / len(span_idx_list)
            span_weight = min(max(span_weight, 1.0), self.span_weight)
        for cno in range(len(cells)):
            if 'bbox' in cells[cno]:
                bbox = cells[cno]['bbox'].copy()
                bbox[0] = bbox[0] * 1.0 / img_width
                bbox[1] = bbox[1] * 1.0 / img_height
                bbox[2] = bbox[2] * 1.0 / img_width
                bbox[3] = bbox[3] * 1.0 / img_height
                td_idx = td_idx_list[cno]
                bbox_list[td_idx] = bbox
                bbox_list_mask[td_idx] = 1.0
                cand_span_idx = td_idx + 1
                if cand_span_idx < (self.max_elem_length + 2):
                    if structure[cand_span_idx] in span_idx_list:
                        structure_mask[cand_span_idx] = span_weight

        data['bbox_list'     ] = bbox_list
        data['bbox_list_mask'] = bbox_list_mask
        data['structure_mask'] = structure_mask
        char_beg_idx = self.get_beg_end_flag_idx('beg', 'char')
        char_end_idx = self.get_beg_end_flag_idx('end', 'char')
        elem_beg_idx = self.get_beg_end_flag_idx('beg', 'elem')
        elem_end_idx = self.get_beg_end_flag_idx('end', 'elem')
        data['sp_tokens'] = np.array([
            char_beg_idx, char_end_idx, 
            elem_beg_idx, elem_end_idx, elem_char_idx1, elem_char_idx2, 
            self.max_text_length, self.max_elem_length, self.max_cell_num, elem_num
        ])
        return data

    def encode(self, text, char_or_elem):
        """
        convert text-label into text-index.
        """
        if char_or_elem == "char":
            max_len = self.max_text_length
            current_dict = self.dict_character
        else:
            max_len = self.max_elem_length
            current_dict = self.dict_elem
        if len(text) > max_len:
            return None
        if len(text) == 0:
            if char_or_elem == "char":
                return [self.dict_character['space']]
            else:
                return None
        text_list = []
        for char in text:
            if char not in current_dict:
                return None
            text_list.append(current_dict[char])
        if len(text_list) == 0:
            if char_or_elem == "char":
                return [self.dict_character['space']]
            else:
                return None
        return text_list

    def get_ignored_tokens(self, char_or_elem):
        beg_idx = self.get_beg_end_flag_idx("beg", char_or_elem)
        end_idx = self.get_beg_end_flag_idx("end", char_or_elem)
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end, char_or_elem):
        if char_or_elem == "char":
            if beg_or_end == "beg":
                idx = np.array(self.dict_character[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict_character[self.end_str])
            else:
                assert False, f"Unsupport type {beg_or_end} in get_beg_end_flag_idx of char"
        elif char_or_elem == "elem":
            if beg_or_end == "beg":
                idx = np.array(self.dict_elem[self.beg_str])
            elif beg_or_end == "end":
                idx = np.array(self.dict_elem[self.end_str])
            else:
                assert False, f"Unsupport type {beg_or_end} in get_beg_end_flag_idx of elem"
        else:
            assert False, f"Unsupport type {char_or_elem} in char_or_elem"
        return idx