from functools import partial
import torch
import torch.nn as nn
from copy import deepcopy

from timm.models import register_model

from braincog.base.node.node import *
from braincog.base.connection.layer import WSConv2d
from braincog.datasets import is_dvs_data
from braincog.model_zoo.base_module import BaseModule

try:
    from torchvision.models.utils import load_state_dict_from_url
except ImportError:
    from torchvision._internally_replaced_utils import load_state_dict_from_url

__all__ = ['SEWResNet', 'sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101',
           'sew_resnet152', 'sew_resnext50_32x4d', 'sew_resnext101_32x8d',
           'sew_wide_resnet50_2', 'sew_wide_resnet101_2']

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


# modified by https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

def sew_function(x: torch.Tensor, y: torch.Tensor, sew_cnf: str):
    if sew_cnf == 'ADD':
        return x + y
    elif sew_cnf == 'AND':
        return x * y
    elif sew_cnf == 'IAND':
        return x * (1. - y)
    else:
        return x + y


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, conv_fun=nn.Conv2d):
    '''3x3 convolution with padding'''
    # print(conv_fun)
    return conv_fun(in_planes,
                    out_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=dilation,
                    groups=groups,
                    bias=False,
                    dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, conv_fun=nn.Conv2d):
    '''1x1 convolution'''
    return conv_fun(in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            groups=1,
            base_width=64,
            dilation=1,
            norm_layer=None,
            sew_cnf: str = None,
            node=torch.nn.Identity,
            conv_fun=nn.Conv2d,
            **kwargs
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, conv_fun=conv_fun)
        self.bn1 = norm_layer(planes)
        self.sn1 = node(channel=planes)
        self.conv2 = conv3x3(planes, planes, conv_fun=conv_fun)
        self.bn2 = norm_layer(planes)
        self.sn2 = node(channel=planes)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = node(channel=planes)
        self.stride = stride
        self.sew_cnf = sew_cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(identity, out, self.sew_cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'sew_cnf={self.sew_cnf}'


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None,
                 sew_cnf: str = None,
                 node=torch.nn.Identity,
                 conv_fun=torch.nn.Conv2d,
                 **kwargs
                 ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, conv_fun=conv_fun)
        self.bn1 = norm_layer(width)
        self.sn1 = node(channel=width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, conv_fun=conv_fun)
        self.bn2 = norm_layer(width)
        self.sn2 = node(channel=width)
        self.conv3 = conv1x1(width, planes * self.expansion, conv_fun=conv_fun)
        self.bn3 = norm_layer(planes * self.expansion)
        self.sn3 = node(channel=planes * self.expansion)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = node(channel=planes * self.expansion)
        self.stride = stride
        self.sew_cnf = sew_cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sn3(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        out = sew_function(out, identity, self.sew_cnf)

        return out

    def extra_repr(self) -> str:
        return super().extra_repr() + f'sew_cnf={self.sew_cnf}'


class SEWResNet(BaseModule):
    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            sew_cnf: str = None,
            step=8,
            node_type=LIFNode,
            encode_type='direct',
            spike_output=False,
            conv_type='normal',
            *args,
            **kwargs
    ):
        super().__init__(
            step,
            encode_type,
            *args,
            **kwargs
        )

        self.spike_output = spike_output
        self.num_classes = num_classes
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        conv_type = conv_type.lower()
        if conv_type == 'normal':
            self.conv_fun = nn.Conv2d
        elif conv_type == 'ws':
            self.conv_fun = WSConv2d

        self.dataset = kwargs['dataset']
        if is_dvs_data(self.dataset):
            init_channel = 2 * self.init_channel_mul
        else:
            init_channel = 3 * self.init_channel_mul

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if self.dataset == 'imnet':
            self.conv1 = nn.Sequential(
                self.conv_fun(init_channel, self.inplanes,
                              kernel_size=7, stride=2, padding=3, bias=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:
            self.conv1 = self.conv_fun(init_channel * self.init_channel_mul, self.inplanes,
                                       kernel_size=3, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = self.node(channel=self.inplanes)

        self.layer1 = self._make_layer(block, 64, layers[0], sew_cnf=sew_cnf)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], sew_cnf=sew_cnf)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], sew_cnf=sew_cnf)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], sew_cnf=sew_cnf)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, sew_cnf: str = None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, conv_fun=self.conv_fun),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            previous_dilation, norm_layer, sew_cnf, node=self.node, conv_fun=self.conv_fun))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, sew_cnf=sew_cnf, node=self.node, conv_fun=self.conv_fun))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def forward(self, inputs):
        if not self.online_training:
            inputs = self.encoder(inputs)
            self.reset()
            if self.layer_by_layer:
                x = self._forward_impl(inputs)
                if self.tet_loss:
                    x = rearrange(x, '(t b) c -> t b c', t=self.step)
                elif not self.temporal_output:
                    x = rearrange(x, '(t b) c -> b c t', t=self.step).mean(-1)
                return x
            else:
                outputs = []
                for t in range(self.step):
                    x = self._forward_impl(inputs[t])
                    outputs.append(x)
                if self.tet_loss:
                    return torch.stack(outputs, dim=0)
                else:
                    if self.temporal_output:
                        return torch.cat(outputs, dim=0)
                    else:
                        return sum(outputs) / len(outputs)
        else:
            x = self._forward_impl(inputs)
            return x


class SEWResNetCIFAR(BaseModule):
    def __init__(
            self,
            block,
            layers,
            num_classes=10,
            zero_init_residual=False,
            groups=1,
            width_per_group=16,
            replace_stride_with_dilation=None,
            norm_layer=None,
            sew_cnf: str = None,
            step=8,
            node_type=LIFNode,
            encode_type='direct',
            spike_output=False,
            conv_type='normal',
            *args,
            **kwargs
    ):
        super().__init__(
            step,
            encode_type,
            *args,
            **kwargs
        )

        self.spike_output = spike_output
        self.num_classes = num_classes
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)

        conv_type = conv_type.lower()
        if conv_type == 'normal':
            self.conv_fun = nn.Conv2d
        elif conv_type == 'ws':
            self.conv_fun = WSConv2d

        self.dataset = kwargs['dataset']
        if is_dvs_data(self.dataset):
            init_channel = 2 * self.init_channel_mul
        else:
            init_channel = 3 * self.init_channel_mul

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = self.conv_fun(init_channel * self.init_channel_mul, self.inplanes,
                                   kernel_size=3, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = self.node(channel=self.inplanes)

        self.layer1 = self._make_layer(block, 16, layers[0], sew_cnf=sew_cnf)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], sew_cnf=sew_cnf)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], sew_cnf=sew_cnf)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, sew_cnf: str = None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, conv_fun=self.conv_fun),
                norm_layer(planes * block.expansion),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                            previous_dilation, norm_layer, sew_cnf, node=self.node, conv_fun=self.conv_fun))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, sew_cnf=sew_cnf, node=self.node, conv_fun=self.conv_fun))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.sn1(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def forward(self, inputs):

        if not self.online_training:
            inputs = self.encoder(inputs)
            self.reset()
            if self.layer_by_layer:
                x = self._forward_impl(inputs)
                if self.tet_loss:
                    x = rearrange(x, '(t b) c -> t b c', t=self.step)
                elif not self.temporal_output:
                    x = rearrange(x, '(t b) c -> b c t', t=self.step).mean(-1)
                return x
            else:
                outputs = []
                for t in range(self.step):
                    x = self._forward_impl(inputs[t])
                    outputs.append(x)
                if self.tet_loss:
                    return torch.stack(outputs, dim=0)
                else:
                    if self.temporal_output:
                        return torch.cat(outputs, dim=0)
                    else:
                        return sum(outputs) / len(outputs)
        else:
            x = self._forward_impl(inputs)
            return x



def _sew_resnet(arch, block, layers, pretrained, progress, sew_cnf, **kwargs):
    model = SEWResNet(block, layers, sew_cnf=sew_cnf, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


@register_model
def sew_resnet18(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    return _sew_resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_resnet34(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    return _sew_resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_resnet50(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    return _sew_resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_resnet101(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    return _sew_resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_resnet152(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    return _sew_resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_resnext50_32x4d(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _sew_resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_resnext101_32x8d(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _sew_resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_wide_resnet50_2(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _sew_resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], pretrained, progress, sew_cnf, **kwargs)


@register_model
def sew_wide_resnet101_2(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _sew_resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], pretrained, progress, sew_cnf, **kwargs)


def _sew_resnet_cifar(arch, block, layers, pretrained, progress, sew_cnf, **kwargs):
    model = SEWResNetCIFAR(block, layers, sew_cnf=sew_cnf, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


@register_model
def sew_resnet20(pretrained=False, progress=True, sew_cnf: str = None, **kwargs):
    return _sew_resnet_cifar('wide_resnet101_2', BasicBlock, [3, 3, 3], pretrained, progress, sew_cnf, **kwargs)
