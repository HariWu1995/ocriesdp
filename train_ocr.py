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


model_dir = Path(__file__).parent
root_dir = model_dir.parent
data_dir = root_dir / 'datasets'

datasets_ocr = [
    'BKAINaver_scene_text_1', 'BKAINaver_scene_text_2',
    'CORD_receipt', 'FUNSD_document_scanned', 'MCOCR_receipt', 'SROIE_invoice',
]

# Load data
from datasets import load_dataset
from models.heads.multitask.PGNet.dataloader import Dataloader

for data_name in datasets_ocr:
    data_type = data_name.split('_')[0]
    print((data_dir / data_name).absolute())
    dataset = load_dataset((data_dir / data_name).absolute(), data_type)
    dataloader = Dataloader(dataset['image'], 
                            dataset['label'], 
                            config_path=(data_dir / f'{data_name}.json').absolute(), 
                            batch_size=16, shuffle=True)
    break

print('\n\n\n Loaded data ...')
for k,v in dataloader[1].items():
    print(k.ljust(19), v.size())

batch_data = dataloader[1]

image_ts = batch_data.pop('image')
labels_ts = batch_data


# Load model
from models.backbones.resnet import build_resnet
from models.necks.fpn_vanilla import FPN
# from models.necks.fpn_panoptic import PanopticFPN as FPN
from models.necks.fpn_panet import PANetFPN
from models.heads.multitask.PGNet.model import PGNet, PGHead
from models.heads.multitask.PGNet.loss import PGLoss


print('\n\n\n Features being extracted ...')
backbone = build_resnet(output_channels=2048, variant_depth=50, load_pretrained=True)
features = backbone(image_ts, output='all')

for fi, feat in enumerate(features):
    print(fi, feat.size())

print('\n\n\n Features being fed to neck ...')
feat_maps = features[1:4]
feat_dim = 256
neck1 = FPN(in_feats_shapes=[                      feat.size()      for feat in feat_maps      ], out_channels=feat_dim, hidden_channels=feat_dim)
neck2 = FPN(in_feats_shapes=[[16, feat_dim] + list(feat.size()[2:]) for feat in feat_maps[::-1]], out_channels=feat_dim, hidden_channels=feat_dim)
# neck = neck1 
neck = PANetFPN(neck1, neck2)
features = neck(feat_maps)
for fi, feat in enumerate(features):
    print(fi, feat.size())

print('\n\n\n Features being fed to head ...')
feat_map = features[-1]
head = PGHead(in_channels=feat_dim, num_characters=dataloader.Preprocessor.vocab_size)
outputs = head(feat_map)
for k,v in outputs.items():
    print(k.ljust(11), v.size())

# model = PGNet(num_characters=dataloader.Preprocessor.vocab_size, use_fpn=False)
# outputs = model(features)

print('\n\n\n Calculating loss ...')
loss_fn = PGLoss(pad_num=dataloader.Preprocessor.vocab_size,
                 tcl_len=dataloader.Preprocessor.tcl_len, 
            max_text_len=dataloader.Preprocessor.max_text_len,
          max_bbox_count=dataloader.Preprocessor.max_bbox_count,)
loss = loss_fn(outputs, labels_ts)
print(loss)


