import numpy as np

from src.data.label import RecLabelEncoder


class CTCLabelEncoder(RecLabelEncoder):
    """ 
    Convert between text-label and text-index 
    """
    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelEncoder, self).__init__(max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
            
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data['label_ace'] = np.array(label)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character