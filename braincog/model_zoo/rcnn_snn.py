# from tkinter.messagebox import NO
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F
from timm.models import register_model
from braincog.model_zoo.base_module import BaseModule
from einops import rearrange
from braincog.base.node.node import *


# adapted from
# https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = width_per_group
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base_width, layers[0])
        self.layer2 = self._make_layer(block, self.base_width * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.base_width * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, self.base_width * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.base_width * 8 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=16, replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = width_per_group
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base_width, layers[0])
        self.layer2 = self._make_layer(block, self.base_width * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, self.base_width * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.base_width * 4 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

    def reset(self):
        for i in self.modules():
            if hasattr(i, "n_reset"):
                i.n_reset()


'''
def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
'''


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet_6n2_cifar(n=3, **kwargs):
    r"""ResNet-20 model for CIFAR10 from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return ResNetCIFAR(BasicBlock, [n, n, n], **kwargs)


class RCL(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(RCL, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps

        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x
        for i in range(self.steps):
            if i == 0:
                z = self.conv(x)
            else:
                z = self.conv(x) + self.shortcut(rx)

            x = self.bn[i](z)
            x = self.relu(x)
        return x


class SpikingRCL(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCL, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
        else:
            z = self.conv(self.lastresult) + self.shortcut(rx)

        x = self.bn[self.i](z)
        x = self.relu(x)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCL2(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCL2, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.conv(x)
        else:
            z = self.conv(self.lastresult) + self.shortcut(rx)

        x = self.bn[self.i](z)
        x = self.relu(x)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCLbn(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbn, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)

        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCLbn3(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbn3, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            z = self.bn2[self.i](z)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)

        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCLbn6(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbn6, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            z = self.bn2[self.i](z)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)

        x = self.bn[0](z)
        x = self.relu(z)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCLbn2(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbn2, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[0](z2)

        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCLbn2_2(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbn2_2, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            z = self.bn2[0](z)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[0](z2)

        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCLbn2_3(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbn2_3, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            # z=self.bn2[0](z)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[0](z2)

        z = self.bn2[0](z)
        x = self.relu(z)
        self.lastresult = x
        self.i = self.i + 1
        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None


class SpikingRCLbnadd(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnadd, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.lastresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
        else:
            z1 = self.conv(self.lastresult[-1])
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)

        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult.append(x)
        self.i = self.i + 1

        x = torch.stack(self.lastresult).sum(0)

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = []


class SpikingRCLbnadd2(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnadd2, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn22 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])

        self.bng = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bng2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.steps > 1: self.shortcutg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1,
                                                      bias=False)

        self.lastresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            g = 0
        else:

            g1 = self.convg(self.lastresult[-1])
            g2 = self.shortcutg(rx)
            g = self.bng[self.i](g1) + self.bng2[self.i](g2)

            g = torch.sigmoid(g)
            z1 = self.conv(self.lastresult[-1])
            z2 = self.shortcut(rx)
            z = self.bn22[self.i](self.bn[self.i](z1) * g) + self.bn2[self.i](z2)
        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult.append(x)
        self.i = self.i + 1

        # x=torch.stack(self.lastresult).sum(0)

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = []


class SpikingRCLbnadd2_3(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnadd2_3, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn22 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])

        self.bng = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bng2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.steps > 1: self.shortcutg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1,
                                                      bias=False)

        self.lastresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            z = self.bn2[self.i](z)
            g = 0
        else:

            g1 = self.convg(self.lastresult[-1])
            g2 = self.shortcutg(rx)
            g = self.bng[self.i](g1) + self.bng2[self.i](g2)

            g = torch.sigmoid(g)
            z1 = self.conv(self.lastresult[-1])
            z2 = self.shortcut(rx)
            z = self.bn22[self.i](self.bn[self.i](z1) * g) + self.bn2[self.i](z2)
        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult.append(x)
        self.i = self.i + 1

        # x=torch.stack(self.lastresult).sum(0)

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = []


class SpikingRCLbnadd2_2(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnadd2_2, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn22 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])

        self.bng = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bng2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.steps > 1: self.shortcutg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1,
                                                      bias=False)

        self.lastresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            g = 0
        else:

            g1 = self.convg(self.lastresult[-1])
            g2 = self.shortcutg(rx)
            g = self.bng[self.i](g1) + self.bng2[self.i](g2)

            g = torch.sigmoid(g)
            z1 = self.conv(self.lastresult[-1])
            z2 = self.shortcut(rx)
            z = self.bn22[self.i](self.bn[self.i](z1) * g) + self.bn2[self.i](z2)
        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult.append(x.detach())
        self.i = self.i + 1

        # x=torch.stack(self.lastresult).sum(0)

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = []


class SpikingRCLbnadd3(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnadd3, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn22 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])

        self.bng = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bng2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.steps > 1: self.shortcutg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1,
                                                      bias=False)

        self.lastresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            g = 1
        else:

            g1 = self.convg(self.lastresult[-1])
            g2 = self.shortcutg(rx)
            g = self.bng[self.i](g1) + self.bng2[self.i](g2)

            g = torch.sigmoid(g)

            z1 = self.conv(self.lastresult[-1])
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)
        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult.append(x * g)
        self.i = self.i + 1

        x = torch.stack(self.lastresult).sum(0)

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = []


class SpikingRCLbnadd5(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnadd5, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn22 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])

        self.bng = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bng2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.steps > 1: self.shortcutg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1,
                                                      bias=False)

        self.lastresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            g = self.shortcutg(x)
            g = torch.sigmoid(g)
        else:

            g1 = self.convg(self.lastresult[-1])
            g2 = self.shortcutg(rx)
            g = self.bng[self.i](g1) + self.bng2[self.i](g2)

            g = torch.sigmoid(g)

            z1 = self.conv(self.lastresult[-1])
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)
        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult.append(x * g)
        self.i = self.i + 1

        x = torch.stack(self.lastresult).sum(0)

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = []


class SpikingRCLbnadd4(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnadd4, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.convg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn22 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])

        self.bng = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bng2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        if self.steps > 1: self.shortcutg = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1,
                                                      bias=False)

        self.lastresult = []
        self.lastg = []
        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
            g = 1
        else:

            g1 = self.convg(self.lastresult[-1])
            g2 = self.shortcutg(rx)
            g = self.bng[self.i](g1) + self.bng2[self.i](g2)

            g = torch.sigmoid(g)

            z1 = self.conv(self.lastresult[-1])
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)
        # x = self.bn[self.i](z)
        x = self.relu(z)
        self.lastresult.append(x)
        self.lastg.append(x * g)
        self.i = self.i + 1

        x = torch.stack(self.lastresult).sum(0)

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = []
        self.lastg = []


class SpikingRCLbnbdd(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnbdd, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.recordresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)

        # x = self.bn[self.i](z)
        self.recordresult.append(z)
        z = torch.stack(self.recordresult).sum(0)
        x = self.relu(z)
        self.lastresult = x
        self.i = self.i + 1

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None
        self.recordresult = []


class SpikingRCLbnbdd2(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnbdd2, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.recordresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)

        self.recordresult.append(z)
        z = torch.stack(self.recordresult).sum(0)
        z = self.bn3(z)
        x = self.relu(z)
        self.lastresult = z
        self.i = self.i + 1

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None
        self.recordresult = []


class SpikingRCLbnbdd3(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLbnbdd3, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(inplanes) for i in range(steps)])
        self.relu = nn.ReLU(inplace=True)
        self.steps = steps
        self.i = 0
        if self.steps > 1: self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.recordresult = []

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        rx = x

        if self.i == 0:
            z = self.shortcut(x)
        else:
            z1 = self.conv(self.lastresult)
            z2 = self.shortcut(rx)
            z = self.bn[self.i](z1) + self.bn2[self.i](z2)

        self.recordresult.append(z)
        z = torch.stack(self.recordresult).sum(0)
        z = self.bn3[self.i](z)
        x = self.relu(z)
        self.lastresult = z
        self.i = self.i + 1

        return x

    def n_reset(self):
        self.i = 0
        self.lastresult = None
        self.recordresult = []


class SpikingRCLconv(nn.Module):
    def __init__(self, inplanes, steps=4):
        super(SpikingRCLconv, self).__init__()
        self.conv = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(inplanes)
        self.relu = LIFNode(threshold=1., layer_by_layer=True)
        self.steps = steps
        self.i = 0
        # if self.steps>1:self.shortcut = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class simpleconv(nn.Module):
    def __init__(self, channels, num_classes, K=96, steps=4):
        super(simpleconv, self).__init__()
        self.K = K

        self.layer1 = nn.Conv2d(channels, K, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(K)
        self.pooling1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pooling2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = RCL(K, steps=steps)
        self.layer3 = RCL(K, steps=steps)
        self.layer4 = RCL(K, steps=steps)
        self.layer5 = RCL(K, steps=steps)

        self.fc = nn.Linear(K, num_classes, bias=True)
        self.dropout = nn.Dropout(p=0.5)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn(self.relu(x))
        x = self.pooling1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pooling2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.max_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K)
        x = self.dropout(x)
        x = self.fc(x)
        return x


@register_model
class RCNN(BaseModule):
    def __init__(self, channels=2, num_classes=11, K=96, step=10, RCL=RCL, **kwargs):
        super(RCNN, self).__init__(step=step, encode_type='direct', layer_by_layer=True)
        self.K = K
        self.layer1 = nn.Conv2d(channels, K, kernel_size=3, padding=1)
        self.relu = LIFNode(threshold=1., layer_by_layer=True)
        self.bn = nn.BatchNorm2d(K)
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = SpikingRCLconv(K, steps=self.step)
        self.layer3 = SpikingRCLconv(K, steps=self.step)
        self.layer4 = SpikingRCLconv(K, steps=self.step)
        self.layer5 = SpikingRCLconv(K, steps=self.step)

        self.fc = nn.Linear(K, num_classes, bias=True)
        # self.dropout = nn.Dropout(p=0.5)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        self.reset()

        x = self.encoder(x)
        if self.layer_by_layer:
            x = self.layer1(x)
            # x = self.bn(self.relu(x))

            x = self.relu(self.bn(x))
            x = self.pooling1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.pooling2(x)
            x = self.layer4(x)
            x = self.layer5(x)
            x = F.avg_pool2d(x, x.shape[-1])
            x = torch.flatten(x, 1)
            # x = self.dropout(x)
            x = self.fc(x)
            x = rearrange(x, '(t b) c -> b c t', t=self.step)
            x = x.mean(-1)
            # print(x.shape)
        return x

    def reset(self):
        for i in self.modules():
            if hasattr(i, "n_reset"):
                i.n_reset()


class RCNNmnist(nn.Module):
    def __init__(self, channels, num_classes, K=96, steps=4, RCL=RCL):
        super(RCNNmnist, self).__init__()
        self.K = K
        self.steps = steps
        self.layer1 = nn.Conv2d(channels, K, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(K)
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = RCL(K, steps=steps)
        self.layer3 = RCL(K, steps=steps)
        self.layer4 = RCL(K, steps=steps)
        self.layer5 = RCL(K, steps=steps)

        self.fc = nn.Linear(K, num_classes, bias=True)
        self.dropout = nn.Dropout(p=0.5)

        # init the parameter
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.layer1(x)
        # x = self.bn(self.relu(x))

        x = self.relu(self.bn(x))
        x = self.pooling1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pooling2(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(-1, self.K)
        # x = self.dropout(x)
        x = self.fc(x)
        return x

    def reset(self):
        for i in self.modules():
            if hasattr(i, "n_reset"):
                i.n_reset()
