import numpy as np

from data.label import RecLabelEncoder


class SEEDLabelEncoder(RecLabelEncoder):
    """ 
    SEED: Semantics Enhanced Encoder-Decoder Framework for Scene Text Recognition
    """
    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs):
        super(SEEDLabelEncoder, self).__init__(
            max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.padding = "padding"
        self.end_str = "eos"
        self.unknown = "unknown"
        dict_character = dict_character + [self.end_str, self.padding, self.unknown]
        return dict_character

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data['length'] = np.array(len(text)) + 1  # conclude eos
        text = text + [len(self.character) - 3] \
                    + [len(self.character) - 2] * (self.max_text_len - len(text) - 1)
        data['label'] = np.array(text)
        return data


