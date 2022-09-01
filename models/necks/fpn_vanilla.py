from typing import Tuple, Sequence, Optional, Iterable
from torch import nn
from models.necks.layers import Reverse, Sum, SequentialMultiInputMultiOutput, Parallel, Interpolate


class FPN(nn.Sequential):
    """
    "Feature Pyramid Networks for Object Detection" by Lin et al., 
        https://arxiv.com/abs/1612.03144.
    
    Takes in an n-tuple of feature maps in reverse order (1st feature map, 2nd feature map, ..., nth feature map), 
    where 1st feature map is the one produced by the earliest layer in the backbone network.
    The feature maps are passed through the architecture shown below, producing n outputs, 
    such that the size the ith output is equal to the corresponding input feature map 
    and the number of channels is equal to out_channels.
    
    Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)
    
    Architecture diagram:
    nth feat. map ────────[nth in_conv]──────────┐────────[nth out_conv]────> nth out
                                                 │
                                             [upsample]
                                                 │
                                                 V
    (n-1)th feat. map ────[(n-1)th in_conv]────>(+)────[(n-1)th out_conv]────> (n-1)th out
                                                 │
                                             [upsample]
                                                 │
                                                 V
            .                     .                           .                    .
            .                     .                           .                    .
            .                     .                           .                    .
                                                 │
                                             [upsample]
                                                 │
                                                 V
    1st feat. map ────────[1st in_conv]────────>(+)────────[1st out_conv]────> 1st out
    """
    def __init__(self, in_feats_shapes: Sequence[Tuple[int, ...]], hidden_channels: int = 256, out_channels: int = 2):

        # reverse so that the deepest (i.e. produced by the deepest layer in
        # the backbone network) feature map is first.
        in_feats_shapes = in_feats_shapes[::-1]
        in_feats_channels = [s[1] for s in in_feats_shapes]

        # 1x1 conv to make the channels of all feature maps the same
        in_convs = Parallel([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1) for in_channels in in_feats_channels
        ])

        # yapf: disable
        def resize_and_add(to_size):
            return nn.Sequential(Parallel([nn.Identity(), Interpolate(size=to_size)]), Sum())

        top_down_layer = SequentialMultiInputMultiOutput(
            nn.Identity(),
            *[resize_and_add(shape[-2:]) for shape in in_feats_shapes[1:]]
        )

        out_convs = Parallel([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1) for _ in in_feats_shapes
        ])
        
        layers = [Reverse(), in_convs, top_down_layer, out_convs, Reverse()]
        # yapf: enable
        super().__init__(*layers)


