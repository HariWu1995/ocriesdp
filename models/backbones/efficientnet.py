# Github: https://github.com/lukemelas/EfficientNet-PyTorch
# With adjustments and added comments by workingcoder (github username).

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

from models.activations import Swish, SwishMemoryEfficient
from models.backbones.efficientnet_utils import (
    round_filters, round_repeats, drop_connect,
    get_conv2d_same_padding, get_maxPool2d_same_padding,
    get_model_params, efficientnet_params, calculate_output_image_size,
)


VALID_MODELS = [f'b{i}' for i in range(9)] + ['l2'] # l2: pretrain unavailable


class Bottleneck(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block.
    
    Args:
        block_params (namedtuple): BlockParams, defined in utils/preprocess/efficientnet.py.
        global_params (namedtuple): GlobalParam, defined in utils/preprocess/efficientnet.py.
        image_size (tuple or list): [image_height, image_width].
    References:
        [1] https://arxiv.org/abs/1704.04861 (MobileNet v1)
        [2] https://arxiv.org/abs/1801.04381 (MobileNet v2)
        [3] https://arxiv.org/abs/1905.02244 (MobileNet v3)
    """

    def __init__(self, block_params, global_params, image_size=None):
        super().__init__()
        self.block_params = block_params
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self.block_params.se_ratio is not None) and (0 < self.block_params.se_ratio <= 1)
        self.id_skip = block_params.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self.block_params.input_filters                                   # number of input channels
        oup = self.block_params.input_filters * self.block_params.expand_ratio  # number of output channels
        if self.block_params.expand_ratio != 1:
            Conv2d = get_conv2d_same_padding(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depth-wise convolution phase
        k = self.block_params.kernel_size
        s = self.block_params.stride
        Conv2d = get_conv2d_same_padding(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze-and-Excitation layer, if desired
        if self.has_se:
            Conv2d = get_conv2d_same_padding(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self.block_params.input_filters * self.block_params.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Point-wise convolution phase
        final_oup = self.block_params.output_filters
        Conv2d = get_conv2d_same_padding(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = SwishMemoryEfficient()

    def forward(self, inputs, drop_connect_rate=None):

        # Expansion and Depth-wise Convolution
        x = inputs
        if self.block_params.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze-and-Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Point-wise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters = self.block_params.input_filters
        output_filters = self.block_params.output_filters
        if self.id_skip and self.block_params.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient: bool = True):
        """
        Sets swish function as memory efficient (for training) or standard (for export).
        
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = SwishMemoryEfficient() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    EfficientNet model.
    Most easily loaded with the .from_name or .from_pretrained methods.
    
    Args:
        blocks_params (list[namedtuple]): A list of BlockParams to construct blocks.
        global_params (     namedtuple ): A set of GlobalParams shared between blocks.
    
    References:
        [1] https://arxiv.org/abs/1905.11946 (EfficientNet)
    
    Example:
    >>> import torch
    >>> from models.backbones.efficientnet import EfficientNet
    >>> inputs = torch.rand(1, 3, 224, 224)
    >>> model = EfficientNet.from_pretrained('b0')
    >>> model.eval()
    >>> outputs = model(inputs)
    """
    def __init__(self, blocks_params=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_params, list), 'blocks_params should be a list'
        assert len(blocks_params) > 0, 'size of blocks_params must be greater than 0'
        self._global_params = global_params
        self._blocks_params = blocks_params

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps =     self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = global_params.image_size
        Conv2d = get_conv2d_same_padding(image_size=image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_params in self._blocks_params:

            # Update block input and output filters based on depth multiplier.
            block_params = block_params._replace(
                 input_filters=round_filters(block_params.input_filters,  self._global_params),
                output_filters=round_filters(block_params.output_filters, self._global_params),
                    num_repeat=round_repeats(block_params.num_repeat,     self._global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(Bottleneck(block_params, self._global_params, image_size=image_size))
            image_size = calculate_output_image_size(image_size, block_params.stride)
            if block_params.num_repeat > 1:  # modify block_params to keep same output size
                block_params = block_params._replace(input_filters=block_params.output_filters, stride=1)
            for _ in range(1, block_params.num_repeat):
                self._blocks.append(Bottleneck(block_params, self._global_params, image_size=image_size))
                # image_size = calculate_output_image_size(image_size, block_params.stride)  # stride = 1

        # Head
        in_channels = block_params.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_conv2d_same_padding(image_size=image_size)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)

        # set activation to memory efficient swish by default
        self._swish = SwishMemoryEfficient()

    def set_swish(self, memory_efficient: bool = True):
        """
        Sets swish function as memory efficient (for training) or standard (for export).
        
        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = SwishMemoryEfficient() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """
        Use convolution layer to extract features from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.
        
        Returns:
            Dictionary of last intermediate features with reduction levels i in [1, 2, 3, 4, 5].

        Example:
        >>> import torch
        >>> from models.backbones.efficientnet import EfficientNet
        >>> inputs = torch.rand(1, 3, 224, 224)
        >>> model = EfficientNet.from_pretrained('b0')
        >>> endpoints = model.extract_endpoints(inputs)
        >>> print(endpoints['reduction_1'].shape)  # torch.Size([1,   16, 112, 112])
        >>> print(endpoints['reduction_2'].shape)  # torch.Size([1,   24,  56,  56])
        >>> print(endpoints['reduction_3'].shape)  # torch.Size([1,   40,  28,  28])
        >>> print(endpoints['reduction_4'].shape)  # torch.Size([1,  112,  14,  14])
        >>> print(endpoints['reduction_5'].shape)  # torch.Size([1,  320,   7,   7])
        >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280,   7,   7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints

    def extract_features(self, inputs, return_all: bool = False):
        """
        Use convolution layer to extract featuremap.

        Args:
            inputs (tensor): Input tensor.
        
        Returns:
            Output of the final convolution layer in the efficientnet model.
        """
        features = []

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        features.append(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            features.append(x)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        features.append(x)

        return features if return_all else x

    def forward(self, inputs):
        # Convolution layers
        x = self.extract_features(inputs, return_all=False)

        # Pooling
        x = self._avg_pooling(x)

        # Classification
        if self._global_params.include_top:
            x = x.flatten(start_dim=1)
            x = self._dropout(x)
            x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, in_channels=3, **override_params):
        """
        Create an efficientnet model according to name.
        
        Args:
            model_name (str): Name for efficientnet.
            in_channels (int): Input data's channel number.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        
        Returns:
            An efficientnet model.
        """
        cls._check_model_name_is_valid(model_name)
        blocks_params, global_params = get_model_params(model_name, override_params)
        model = cls(blocks_params, global_params)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def from_pretrained(cls, model_name, weights_path=None, advprop=False, in_channels=3, num_classes=1000, **override_params):
        """
        Create an efficientnet model according to name.
        
        Args:
            model_name (str): Name for efficientnet.
            weights_path (None or str):
                str: path to pretrained weights file on the local disk.
                None: use pretrained weights downloaded from the Internet.
            advprop (bool):
                Whether to load pretrained weights
                trained with advprop (valid when weights_path is None).
            in_channels (int): Input data's channel number.
            num_classes (int):
                Number of categories for classification.
                It controls the output size for final linear layer.
            override_params (other key word params):
                Params to override model's global_params.
                Optional key:
                    'width_coefficient', 'depth_coefficient',
                    'image_size', 'dropout_rate',
                    'batch_norm_momentum',
                    'batch_norm_epsilon', 'drop_connect_rate',
                    'depth_divisor', 'min_depth'
        
        Returns:
            A pretrained efficientnet model.
        """
        model = cls.from_name(model_name, num_classes=num_classes, **override_params)
        load_pretrained_weights(model, model_name, weights_path=weights_path,
                                load_fc=(num_classes == 1000), advprop=advprop)
        model._change_in_channels(in_channels)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        """
        Get the input image size for a given efficientnet model.
        
        Args:
            model_name (str): Name for efficientnet.
        
        Returns:
            Input image size (resolution).
        """
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """
        Validates model name.
        
        Args:
            model_name (str): Name for efficientnet.
        
        Returns:
            bool: Is a valid name or not.
        """
        if model_name not in VALID_MODELS:
            raise ValueError('model_name should be one of: ' + ', '.join(VALID_MODELS))

    def _change_in_channels(self, in_channels):
        """
        Adjust model's first convolution layer to in_channels, if in_channels not equals 3.
        
        Args:
            in_channels (int): Input data's channel number.
        """
        if in_channels != 3:
            Conv2d = get_conv2d_same_padding(image_size=self._global_params.image_size)
            out_channels = round_filters(32, self._global_params)
            self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)


# train with Standard methods
# check more details in paper(EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks)
url_map = {
    'b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth',
    'b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth',
    'b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth',
    'b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth',
    'b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth',
    'b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth',
    'b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth',
    'b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth',
}

# train with Adversarial Examples (AdvProp)
# check more details in paper (Adversarial Examples Improve Image Recognition)
url_map_advprop = {
    'b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
    'b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
    'b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
    'b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
    'b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',
    'b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',
    'b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',
    'b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',
    'b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',
}

# TODO: add the petrained weights url map of 'efficientnet-l2'


def load_pretrained_weights(model, model_name, weights_path=None, load_fc=True, advprop=False, verbose=True):
    """
    Loads pretrained weights from weights path or download using url.
    
    Args:
        model (Module): The whole model of efficientnet.
        model_name (str): Model name of efficientnet.
        weights_path (None or str):
            str: path to pretrained weights file on the local disk.
            None: use pretrained weights downloaded from the Internet.
        load_fc (bool): Whether to load pretrained weights for classification layer at the end of the model.
        advprop (bool): Whether to load pretrained weights trained with Adversarial Example (valid when weights_path is None).
    """
    if isinstance(weights_path, str):
        state_dict = torch.load(weights_path)
    else:
        # AutoAugment or Advprop (different preprocessing)
        url_map_ = url_map_advprop if advprop else url_map
        state_dict = model_zoo.load_url(url_map_[model_name])

    if load_fc:
        ret = model.load_state_dict(state_dict, strict=False)
        assert not ret.missing_keys, \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)
    else:
        state_dict.pop('_fc.weight')
        state_dict.pop('_fc.bias')
        ret = model.load_state_dict(state_dict, strict=False)
        assert set(ret.missing_keys) == set(['_fc.weight', '_fc.bias']), \
            'Missing keys when loading pretrained weights: {}'.format(ret.missing_keys)

    assert not ret.unexpected_keys, 'Missing keys when loading pretrained weights: {}'.format(ret.unexpected_keys)

    if verbose:
        print('Loaded pretrained weights for {}'.format(model_name))


