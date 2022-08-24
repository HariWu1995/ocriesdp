from collections import namedtuple
from functools import partial
import re
import math
import torch

from src.models.layers import (Conv2dDynamicSamePadding, MaxPool2dDynamicSamePadding,
                               Conv2dStaticSamePadding, MaxPool2dStaticSamePadding, )

# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = namedtuple('GlobalParams', ['width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                                           'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
                                           'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top',])

# Parameters for an individual model block
BlockParams = namedtuple('BlockParams', ['num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                                         'input_filters', 'output_filters', 'se_ratio', 'id_skip',])

# Set GlobalParams and BlockParams's defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockParams.__new__.__defaults__ = (None,) * len(BlockParams._fields)


def round_filters(filters, global_params):
    """
    Calculate and round number of filters based on width multiplier.
    Use width_coefficient, depth_divisor and min_depth of global_params.
    
    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.
    
    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    
    # TODO: modify the params names.
    # maybe the names (width_divisor, min_width) are more suitable than (depth_divisor, min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth

    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """
    Calculate module's repeat number of a block based on depth multiplier.
    Use depth_coefficient of global_params.
    
    Args:
        repeats (int): num_repeat to be calculated.
        global_params (namedtuple): Global params of the model.
    
    Returns:
        new repeat: New repeat number after calculating.
    """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    # follow the formula transferred from official TensorFlow implementation
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """
    Drop connect.
    
    Args:
        input (tensor: BCWH): Input of this structure.
        p (float): Probability of drop connection.
        training (bool): The running mode.
    
    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """
    Obtain height and width from x.
    
    Args:
        x (int, tuple or list): Data size.
    
    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """
    Calculates the output image size when using Conv2dSamePadding with a stride.
    Necessary for static padding. Thanks to mannatsingh for pointing this out.
    
    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.
    
    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


# Note:
# The following 'SamePadding' functions make output size equal ceil(input size/stride).
# Only when stride equals 1, can the output size be the same as input size.
# Don't be confused by their function names !!!

def get_conv2d_same_padding(image_size=None):
    """
    Chooses static padding if you have specified an image size, and dynamic padding otherwise.
    Static padding is necessary for ONNX exporting of models.
    
    Args:
        image_size (int or tuple): Size of the image.
    
    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


def get_maxPool2d_same_padding(image_size=None):
    """
    Chooses static padding if you have specified an image size, and dynamic padding otherwise.
    Static padding is necessary for ONNX exporting of models.
    
    Args:
        image_size (int or tuple): Size of the image.
    
    Returns:
        MaxPool2dDynamicSamePadding or MaxPool2dStaticSamePadding.
    """
    if image_size is None:
        return MaxPool2dDynamicSamePadding
    else:
        return partial(MaxPool2dStaticSamePadding, image_size=image_size)


class BlockDecoder(object):
    """
    Block Decoder for readability, straight from the official TensorFlow repository.
    """
    @staticmethod
    def _decode_block_string(block_string):
        """
        Get a block through a string notation of arguments.
        
        Args:
            block_string (str): A string notation of arguments.
                                Examples: 'r1_k3_s11_e1_i32_o16_se0.25_noskip'.
        
        Returns:
            BlockArgs: The namedtuple defined at the top of this file.
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockParams(num_repeat =   int(options['r']),
                          kernel_size =   int(options['k']),
                               stride =  [int(options['s'][0])],
                         expand_ratio =   int(options['e']),
                        input_filters =   int(options['i']),
                       output_filters =   int(options['o']),
                             se_ratio = float(options['se']) if 'se' in options else None,
                              id_skip = ('noskip' not in block_string)
        )

    @staticmethod
    def _encode_block_string(block):
        """
        Encode a block to a string.
        
        Args:
            block (namedtuple): A BlockArgs type argument.
        
        Returns:
            block_string: A String form of BlockArgs.
        """
        args = [
              'r%d' %  block.num_repeat,
              'k%d' %  block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
              'e%s' %  block.expand_ratio,
              'i%d' %  block.input_filters,
              'o%d' %  block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decode a list of string notations to specify blocks inside the network.
        
        Args:
            string_list (list[str]): A list of strings, each string is a notation of block.
        
        Returns:
            blocks_args: A list of BlockArgs namedtuples of block args.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encode a list of BlockArgs to a list of strings.
        
        Args:
            blocks_args (list[namedtuples]): A list of BlockArgs namedtuples of block args.
        
        Returns:
            block_strings: A list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_params(model_name):
    """
    Map EfficientNet model name to parameter coefficients.
    
    Args:
        model_name (str): Model name to be queried.
    
    Returns:
        params_dict[model_name]: A (width,depth,res,dropout) tuple.
    """
    params_dict = {
        # Coefficients: width, depth, resolution, dropout
        'b0': (1.0, 1.0, 224, 0.2),
        'b1': (1.0, 1.1, 240, 0.2),
        'b2': (1.1, 1.2, 260, 0.3),
        'b3': (1.2, 1.4, 300, 0.3),
        'b4': (1.4, 1.8, 380, 0.4),
        'b5': (1.6, 2.2, 456, 0.4),
        'b6': (1.8, 2.6, 528, 0.5),
        'b7': (2.0, 3.1, 600, 0.5),
        'b8': (2.2, 3.6, 672, 0.5),
        'l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]


def efficientnet(width_coefficient=None, depth_coefficient=None, image_size=None,
                 dropout_rate=0.2, drop_connect_rate=0.2, num_classes=1000, include_top=True):
    """
    Create BlockArgs and GlobalParams for efficientnet model.
    
    Args:
        width_coefficient (float)
        depth_coefficient (float)
        image_size (int)
        dropout_rate (float)
        drop_connect_rate (float)
        num_classes (int)
        Meaning as the name suggests.
    
    Returns:
        blocks_params, global_params.
    """

    # Blocks args for the whole model(efficientnet-b0 by default)
    # It will be modified in the construction of EfficientNet Class according to model
    blocks_params = [
        'r1_k3_s11_e1_i32_o16_se0.25',
        'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25',
        'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25',
        'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_params = BlockDecoder.decode(blocks_params)

    global_params = GlobalParams(width_coefficient=width_coefficient,
                                 depth_coefficient=depth_coefficient,
                                        image_size=image_size,
                                      dropout_rate=dropout_rate,

                                       num_classes=num_classes,
                               batch_norm_momentum=0.99,
                                batch_norm_epsilon=1e-3,
                                 drop_connect_rate=drop_connect_rate,
                                     depth_divisor=8,
                                         min_depth=None,
                                       include_top=include_top,)

    return blocks_params, global_params


def get_model_params(model_name, override_params):
    """
    Get the block args and global params for a given model name.
    
    Args:
        model_name (str): Model's name.
        override_params (dict): A dict to modify global_params.
    
    Returns:
        blocks_params, global_params
    """
    if (model_name.startswith('b') and model_name[-1].isdigit()) or (model_name == 'l2'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_params, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: {}'.format(model_name))
    
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_params, global_params







