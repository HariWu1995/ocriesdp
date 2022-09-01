from torch import nn
from models.necks.layers import Reverse


class PANetFPN(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Path Aggregation Network for Instance Segmentation" by Liu et al.,
        https://arxiv.com/abs/1803.01534. 
    
    This architecture adds a bottom-up path after the top-down path in a normal FPN. 
    It can be thought of as a normal FPN followed by a flipped FPN.

    Takes in an n-tuple of feature maps in reverse order (1st feature map, 2nd feature map, ..., nth feature map), 
    where the 1st feature map is the one produced by the earliest layer in the backbone network.
    
    The feature maps are passed through the architecture shown below, producing n outputs, 
    such that the size of the ith output is equal to the corresponding input feature map 
    and the number of channels is equal to out_channels.

    Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)
    
    Architecture diagram:
            (1st feature map, 2nd feature map, ..., nth feature map)
                                    │
                                [1st FPN]
                                    │
                                    V
                                    │
                        [Reverse the order of outputs]
                                    │
                                    V
                                    │
                                [2nd FPN]
                                    │
                                    V
                                    │
                        [Reverse the order of outputs]
                                    │
                                    │
                                    V
                       (1st out, 2nd out, ..., nth out)
    """
    def __init__(self, fpn1: nn.Module, fpn2: nn.Module):
        # yapf: disable
        layers = [fpn1, Reverse(), fpn2, Reverse(),]
        
        # yapf: enable
        super().__init__(*layers)

