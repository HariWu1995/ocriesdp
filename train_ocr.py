import os, sys
import io
import json 
import random

from ast import literal_eval as structurize_str
from glob import glob
from tqdm import tqdm

from pathlib import Path
from difflib import SequenceMatcher
from collections import Counter

import cv2
import math
import torch
import numpy as np
import pandas as pd
import seaborn as sns

from PIL import Image


root_dir = Path('.')
data_dir = root_dir / 'datasets'
model_dir = root_dir / 'source_code'

datasets_ocr = [
    'BKAINaver_scene_text_1', 'BKAINaver_scene_text_2',
    'CORD_receipt', 'FUNSD_document_scanned', 'MCOCR_receipt', 'SROIE_invoice',
]

# Load data
from src.data.datasets import load_dataset
from src.models.heads.multitask.PGNet.dataloader import Dataloader

for data_name in datasets_ocr:
    data_type = data_name.split('_')[0]
    print((data_dir / data_name).absolute())
    dataset = load_dataset((data_dir / data_name).absolute(), data_type)
    dataloader = Dataloader(dataset['image'], 
                            dataset['label'], 
                            config_path=(data_dir / f'{data_name}.json').absolute(), shuffle=True)
    break

print('\n\n\n Loaded data ...')
for k,v in dataloader[1].items():
    print(k.ljust(19), v.size())

batch_data = dataloader[1]

image_ts = batch_data.pop('image')
labels_ts = batch_data


# Load model
from src.models.backbones.resnet import build_resnet
from src.models.heads.multitask.PGNet.model import PGNet
from src.models.heads.multitask.PGNet.loss import PGLoss

hidden_dim = 512

print('\n\n\n Features being extracted ...')
backbone = build_resnet(output_channels=hidden_dim, variant_depth=50, load_pretrained=True)
features = backbone(image_ts, output='all')

for fi, feat in enumerate(features):
    print(fi, feat.size())

print('\n\n\n Features being predicted ...')
model = PGNet(in_channels=hidden_dim, num_characters=dataloader.Preprocessor.vocab_size, use_fpn=False)
outputs = model(features)

for k,v in outputs.items():
    print(k.ljust(11), v.size())

print('\n\n\n Calculating loss ...')
loss_fn = PGLoss(pad_num=dataloader.Preprocessor.vocab_size,
                 tcl_len=dataloader.Preprocessor.tcl_len, 
            max_text_len=dataloader.Preprocessor.max_text_len,
          max_bbox_count=dataloader.Preprocessor.max_bbox_count,)
loss = loss_fn(outputs, labels_ts)
print(loss)


