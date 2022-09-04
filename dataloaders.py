from pathlib import Path
from PIL import Image

import math
import cv2
import numpy as np
import pandas as pd
import torch

from datasets import infer_statistics


EPS = 1e-7
CHANNEL_SCALER = {
    2: { 'mean' : 0.485, 'std' : 0.229, },
    1: { 'mean' : 0.456, 'std' : 0.224, },
    0: { 'mean' : 0.406, 'std' : 0.225, },
}


class Dataloader:

    def __init__(self, image_df: pd.DataFrame, bbox_df: pd.DataFrame,
                    config_path: str or Path, batch_size: int = 16, shuffle: bool = False, ):

        image_df = image_df.drop_duplicates(subset=['image_id'])
        self.image_df = image_df.set_index(keys=['image_id'])
        self.bbox_df = bbox_df.set_index(keys=['image_id'])
        self.indices = list(self.image_df.index)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.image_df.shape[0] / self.batch_size)

    def __getitem__(self, index):
        max_index = len(self)
        assert index < max_index, f"index = {index} exceeds {max_index}"
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        data = []
        for image_id in self.indices[start_index:end_index]:
            image_data = {
                'image' : Image.open(self.image_df.loc[image_id, 'image_path']).convert('RGB'),
                'label' : { column : self.bbox_df.loc[image_id, column].values.tolist() 
                        for column in ['polygons', 'text'] },
            }
            if image_data is None:
                continue
            
            data.append({k: v for k, v in image_data.items()})

        # Convert to tensor
        batch_df = pd.DataFrame.from_dict(data)
        
        data_ts = dict()
        for col in ['image']:
            col_arr = np.stack(batch_df[col].values, axis=3).transpose([3,0,1,2])
            col_ts = torch.tensor(col_arr, dtype=torch.float32)
            data_ts[col] = col_ts
        
        return data_ts

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices) # inplace operation




