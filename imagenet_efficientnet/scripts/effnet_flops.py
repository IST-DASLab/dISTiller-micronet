import torch

from torch import nn
from torch.nn import functional as F

from efficientnet_pytorch.utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

from compute_params_flops import _is_not_quantized
from functools import partial
import math
"""
ATTENTION: Most activations are quantized. So non-linearities
work with quantized version of layer outputs. Bias is not quantized.
"""

class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)

class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x, is_not_quantized):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        return (F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups), *ops_conv(x, self, is_not_quantized))


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""
    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x, is_not_quantized):
        x = self.static_padding(x)
        return (F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups), *ops_conv(x, self, is_not_quantized))


def ops_conv(x, layer, is_not_quantized):
    if is_not_quantized:
        multi_add = 1.0
    else:
        multi_add = 4 / 16

    out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
    out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                layer.stride[1] + 1)
    delta_ops = (
        (layer.weight != 0).float().sum() * out_h * out_w / layer.groups * multi_add
       
    )
    delta_ops_total = (
        layer.weight.numel() * out_h * out_w / layer.groups
    )
    if layer.bias is not None:  
        delta_ops += out_h * out_w * layer.bias.size(0)
        delta_ops_total += out_h * out_w * layer.bias.size(0)
    return delta_ops, delta_ops_total

def ops_linear(x, layer, is_not_quantized):
    if is_not_quantized:
        multi_add = 1.0
    else:
        multi_add = 2.4 / 16

    delta_ops = (layer.weight != 0).float().sum() * multi_add + layer.bias.numel()
    delta_ops_total = layer.weight.numel() + layer.bias.numel()
    return delta_ops, delta_ops_total

def ops_non_linearity(x, is_not_quantized):
    if is_not_quantized:
        multi_add = 1.0
    else:
        multi_add = 4.0 / 16

    delta_ops = x.numel() / x.size(0) * multi_add
    delta_ops_total = x.numel() / x.size(0)
    return delta_ops, delta_ops_total

def ops_bn(x, is_not_quantized):
    if is_not_quantized:
        multi_add = 1.0
    else:
        multi_add = 4.0 / 16

    delta_ops_total = x.size(1) * x.size(2) * x.size(3)
    return delta_ops_total * multi_add, delta_ops_total


def ops_adaptive_avg_pool(x, is_not_quantized):
    if is_not_quantized:
        multi_add = 1.0
    else:
        multi_add = 4.0 / 16

    delta_ops = x.size()[1] * x.size()[2] * x.size()[3] * multi_add
    delta_ops_total = x.size()[1] * x.size()[2] * x.size()[3]
    return delta_ops, delta_ops_total


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        ops, total_ops = 0., 0.

        x = inputs
        if self._block_args.expand_ratio != 1:
            x, delta_ops, delta_ops_total = self._expand_conv(inputs, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

            delta_ops, delta_ops_total = ops_bn(x, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
            x = self._bn0(x)

            delta_ops, delta_ops_total = ops_non_linearity(x, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
            x = relu_fn(x)

        x, delta_ops, delta_ops_total = self._depthwise_conv(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

        delta_ops, delta_ops_total = ops_bn(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = self._bn1(x)

        delta_ops, delta_ops_total = ops_non_linearity(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = relu_fn(x)


        if self.has_se:
            delta_ops, delta_ops_total = ops_adaptive_avg_pool(x, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            
            x_squeezed, delta_ops, delta_ops_total = self._se_reduce(x_squeezed, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

            delta_ops, delta_ops_total = ops_non_linearity(x_squeezed, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
            x_squeezed = relu_fn(x_squeezed)

            x_squeezed, delta_ops, delta_ops_total = self._se_expand(x_squeezed, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
            

            delta_ops, delta_ops_total = ops_non_linearity(x, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
            x = torch.sigmoid(x_squeezed) * x

        x, delta_ops, delta_ops_total = self._project_conv(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

        delta_ops, delta_ops_total = ops_bn(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = self._bn2(x)

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection

        delta_ops, delta_ops_total = ops_non_linearity(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        return x, ops, total_ops


class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        ops, total_ops = 0., 0.

        x, delta_ops, delta_ops_total = self._conv_stem(inputs, True)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

        delta_ops, delta_ops_total = ops_bn(x, True)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

        delta_ops, delta_ops_total = ops_non_linearity(x, True)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = relu_fn(x)

        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x, delta_ops, delta_ops_total = block(x, drop_connect_rate=drop_connect_rate)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

        x, delta_ops, delta_ops_total = self._conv_head(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

        delta_ops, delta_ops_total = ops_bn(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = self._bn1(x)

        delta_ops, delta_ops_total = ops_non_linearity(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = relu_fn(x)

        return x, ops, total_ops

    def forward(self, inputs):
        ops, total_ops = 0., 0.
        
        x, delta_ops, delta_ops_total = self.extract_features(inputs)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total

        delta_ops, delta_ops_total = ops_adaptive_avg_pool(x, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        
        if self._dropout:
            delta_ops, delta_ops_total = ops_non_linearity(x, False)
            ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
            x = F.dropout(x, p=self._dropout, training=self.training)

        delta_ops, delta_ops_total = ops_linear(x, self._fc, False)
        ops, total_ops = ops + delta_ops, total_ops + delta_ops_total
        x = self._fc(x)
        return x, ops, total_ops

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))