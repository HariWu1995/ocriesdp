import numpy as np
import torch
from torch import nn

from src.models.backbones.resnet import build_resnet
from src.models.heads.multitask.PGNet.model import PGNet
from src.models.heads.multitask.PGNet.loss import PGLoss


class MultitaskOcrTrainer:

    def __init__(self, multitask_dataloaders: dict, shared_channels: int = 512, 
                       multitask_weights: dict = None, mode: str = 'aggregation', ):
        
        # Choose mode to train multi-task
        if mode.lower().startswith('seq'):
            self.mode = 'sequential'    # backbone weights will be updated after single-task forwards
        else:
            self.mode = 'aggregation'   # backbone weights will be updated after all tasks forward

        # Assign weight to single task when calculating loss
        for k in multitask_dataloaders.keys():
            if k not in multitask_weights.keys():
                multitask_weights[k] = 1.
            elif isinstance(multitask_weights[k], (int, float)):
                multitask_weights[k] = 1.

        # Build Hydra-Net (multi-head model)
        self.backbone = build_resnet(output_channels=shared_channels, variant_depth=50, load_pretrained=True)
        self.heads = nn.ModuleDict()
        self.losses = nn.ModuleDict()
        self.optimizers = nn.ModuleDict()
        
        for k, v in multitask_dataloaders.items():
            self.heads.update({ k: PGNet(in_channels=shared_channels, 
                                    num_characters=v.Preprocessor.vocab_size,), })
            self.losses.update({ k: PGLoss(pad_num=v.Preprocessor.vocab_size,
                                           tcl_len=v.Preprocessor.tcl_len, 
                                      max_text_len=v.Preprocessor.max_text_len,
                                    max_bbox_count=v.Preprocessor.max_bbox_count,), })


