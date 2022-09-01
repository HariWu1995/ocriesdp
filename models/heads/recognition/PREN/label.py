import numpy as np

from data.label import RecLabelEncoder


class PRENLabelEncoder(RecLabelEncoder):
    
    def __init__(self, max_text_length, character_dict_path, use_space_char=False, **kwargs):
        super(PRENLabelEncoder, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        pad_str = '<PAD>'  # 0 
        end_str = '<EOS>'  # 1
        unk_str = '<UNK>'  # 2

        dict_character = [pad_str, end_str, unk_str] + dict_character
        self.pad_idx = 0
        self.end_idx = 1
        self.unk_idx = 2

        return dict_character

    def encode(self, text):
        if len(text) == 0 or len(text) >= self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                text_list.append(self.unk_idx)
            else:
                text_list.append(self.dict[char])
        text_list.append(self.end_idx)
        if len(text_list) < self.max_text_len:
            text_list += [self.pad_idx] * (self.max_text_len - len(text_list))
        return text_list

    def __call__(self, data):
        text = data['label']
        encoded_text = self.encode(text)
        if encoded_text is None:
            return None
        data['label'] = np.array(encoded_text)
        return data