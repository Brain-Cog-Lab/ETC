from functools import partial
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from timm.models import register_model
from einops import rearrange
from braincog.base.node.node import *
from braincog.base.strategy.surrogate import BinaryActivation
from braincog.base.connection.layer import LearnableBias
from braincog.model_zoo.base_module import BaseModule

# stage_out_channel = [64] + [64] * 2 + [128] * 2 + [256] * 3 + [512] * 2  # normal
stage_out_channel = [64] + [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3  # deeper
# stage_out_channel = [64] + [128] * 2 + [256] * 3 + [512] * 8 + [1024] * 3  # -> conv1x1
# stage_out_channel = [128] + [128] * 1 + [128] * 1 + [256] * 1 + [256] * 1  # wider
# BinaryActivation = IFNode


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride, dataset, init_channel_mul):
        super(firstconv3x3, self).__init__()
        if dataset == 'dvsg' or dataset == 'dvsc10' or dataset == 'NCALTECH101' or dataset == 'NCARS' or dataset == 'DVSG':
            self.conv1 = nn.Sequential(
                nn.Conv2d(2 * init_channel_mul, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup)
            )
        elif dataset == 'imnet':
            self.conv1 = nn.Sequential(
                nn.Conv2d(inp * init_channel_mul, oup, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(oup),
                nn.MaxPool2d(2),
            )
        elif dataset == 'cifar10' or dataset == 'cifar100':
            self.conv1 = nn.Sequential(
                nn.Conv2d(inp * init_channel_mul, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup)
            )
        else:
            raise NotImplementedError('Can not recognize dataset: {}'.format(dataset))

    def forward(self, x):
        out = self.conv1(x)
        return out


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, node=BinaryNode):
        super(BasicBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.node = node

        self.move11 = LearnableBias(inplanes)
        self.binary_3x3 = conv3x3(inplanes, inplanes, stride=stride)
        self.bn1 = norm_layer(inplanes)

        # self.se_block1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(),
        #     nn.Linear(inplanes, inplanes // 4),
        #     nn.ReLU(),
        #     nn.Linear(inplanes // 4, inplanes),
        #     nn.Sigmoid(),
        # )
        # self.bn12 = nn.Sequential(
        #     nn.ReLU(),
        #     norm_layer(inplanes)
        # )

        self.move12 = LearnableBias(inplanes)
        self.prelu1 = nn.PReLU(inplanes)
        self.move13 = LearnableBias(inplanes)

        self.move21 = LearnableBias(inplanes)

        if inplanes == planes:
            self.binary_pw = conv3x3(inplanes, planes)
            self.bn2 = norm_layer(planes)

            # self.se_block2 = nn.Sequential(
            #     nn.AdaptiveAvgPool2d(1),
            #     nn.Flatten(),
            #     nn.Linear(inplanes, inplanes // 4),
            #     nn.ReLU(),
            #     nn.Linear(inplanes // 4, inplanes),
            #     nn.Sigmoid(),
            # )
            # self.bn22 = nn.Sequential(
            #     nn.ReLU(),
            #     norm_layer(inplanes)
            # )

        else:
            self.binary_pw_down1 = conv3x3(inplanes, inplanes)
            self.binary_pw_down2 = conv3x3(inplanes, inplanes)
            self.bn2_1 = norm_layer(inplanes)
            self.bn2_2 = norm_layer(inplanes)

            # self.se_block2_1 = nn.Sequential(
            #     nn.AdaptiveAvgPool2d(1),
            #     nn.Flatten(),
            #     nn.Linear(inplanes, inplanes // 4),
            #     nn.ReLU(),
            #     nn.Linear(inplanes // 4, inplanes),
            #     nn.Sigmoid(),
            # )
            # self.bn22_1 = nn.Sequential(
            #     nn.ReLU(),
            #     norm_layer(inplanes)
            # )
            # self.se_block2_2 = nn.Sequential(
            #     nn.AdaptiveAvgPool2d(1),
            #     nn.Flatten(),
            #     nn.Linear(inplanes, inplanes // 4),
            #     nn.ReLU(),
            #     nn.Linear(inplanes // 4, inplanes),
            #     nn.Sigmoid(),
            # )
            # self.bn22_2 = nn.Sequential(
            #     nn.ReLU(),
            #     norm_layer(inplanes)
            # )

        self.move22 = LearnableBias(planes)
        self.prelu2 = nn.PReLU(planes)
        self.move23 = LearnableBias(planes)

        self.binary_activation1 = self.node()
        self.binary_activation2 = self.node()
        self.stride = stride
        self.inplanes = inplanes
        self.planes = planes

        if self.inplanes != self.planes:
            self.pooling = nn.AvgPool2d(2, 2)

    def forward(self, x):

        out1 = self.move11(x)
        # inputs = out1  # for SE Block
        out1 = self.binary_activation1(out1)
        out1 = self.binary_3x3(out1)
        out1 = self.bn1(out1)
        # out1 = out1 * rearrange(self.se_block1(inputs), 'b c -> b c 1 1')  # for SE Block
        # out1 = self.bn12(out1)  # for SE Block

        if self.stride == 2:
            x = self.pooling(x)

        out1 = x + out1

        out1 = self.move12(out1)
        out1 = self.prelu1(out1)
        out1 = self.move13(out1)

        out2 = self.move21(out1)
        # inputs = out2  # for SE Block
        out2 = self.binary_activation2(out2)

        if self.inplanes == self.planes:
            out2 = self.binary_pw(out2)
            out2 = self.bn2(out2)
            # out2 = out2 * rearrange(self.se_block2(inputs), 'b c -> b c 1 1')  # for SE Block
            # out2 = self.bn22(out2)  # for SE Block
            out2 += out1

        else:
            assert self.planes == self.inplanes * 2

            out2_1 = self.binary_pw_down1(out2)
            out2_2 = self.binary_pw_down2(out2)
            out2_1 = self.bn2_1(out2_1)
            out2_2 = self.bn2_2(out2_2)
            # out2_1 = out2_1 * rearrange(self.se_block2_1(inputs), 'b c -> b c 1 1')  # for SE Block
            # out2_1 = self.bn22_1(out2_1)  # for SE Block
            # out2_2 = out2_2 * rearrange(self.se_block2_2(inputs), 'b c -> b c 1 1')  # for SE Block
            # out2_2 = self.bn22_1(out2_2)  # for SE Block
            out2_1 += out1
            out2_2 += out1
            out2 = torch.cat([out2_1, out2_2], dim=1)

        out2 = self.move22(out2)
        out2 = self.prelu2(out2)
        out2 = self.move23(out2)

        return out2


@register_model
class ReactNet(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=IFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super(ReactNet, self).__init__(
            step=step,
            encode_type='direct',
        )
        self.node = node_type
        if issubclass(self.node, BaseNode):
            self.node = partial(self.node, **kwargs, step=step)
        self.dataset = kwargs['dataset']

        self.feature = nn.ModuleList()
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 1, self.dataset, self.init_channel_mul))
            elif stage_out_channel[i - 1] != stage_out_channel[i] and stage_out_channel[i] != 64:
                self.feature.append(BasicBlock(stage_out_channel[i - 1], stage_out_channel[i], 2, self.node))
            else:
                self.feature.append(BasicBlock(stage_out_channel[i - 1], stage_out_channel[i], 1, self.node))
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stage_out_channel[-1], num_classes)

    def forward(self, inputs):
        inputs = self.encoder(inputs)
        self.reset()
        if self.layer_by_layer:
            x = inputs
            for i, block in enumerate(self.feature):
                x = block(x)
            x = self.pool1(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
        else:
            outputs = []
            for t in range(self.step):
                x = inputs[t]
                for i, block in enumerate(self.feature):
                    x = block(x)
                x = self.pool1(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                outputs.append(x)
            return sum(outputs) / self.step
