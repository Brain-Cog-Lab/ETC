from functools import partial
from typing import Union
from torchvision.ops import DeformConv2d
from braincog.base.node.node import *
from braincog.base.connection.layer import *
from braincog.base.encoder.encoder import *


def detect_init_info(**kwargs):  # init-channel, kernel-size, and stride
    dataset = kwargs['dataset'] if 'dataset' in kwargs else ''
    if 'dvs' in dataset or dataset.startswith('N'):
        return 2, 3, 1
    elif 'cifar' in dataset:
        return 3, 3, 1
    elif 'im' in dataset:
        return 4, 3, 4
    else:
        raise ValueError('Cannot recognize {} dataset'.format(dataset))


class DeformConvPack(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 stride,
                 bias,
                 *args,
                 **kwargs):
        super(DeformConvPack, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if isinstance(self.kernel_size, tuple) or isinstance(self.kernel_size, list):
            self.receptive_field = self.kernel_size[0]
        else:
            self.receptive_field = self.kernel_size
            self.kernel_size = (self.kernel_size, self.kernel_size)

        self.receptive_field = 4 * (self.receptive_field // 2)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True)
        self.deform_conv = DeformConv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        stride=stride,
                                        bias=bias)
        self.init_weights()

    def init_weights(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x):
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        offset = self.receptive_field * (torch.sigmoid(offset) - 0.5)
        mask = torch.sigmoid(mask)
        return self.deform_conv(x, offset, mask)


# class MyMaxPool2d(nn.Module):
#     def __init__(self, kernel_size=2, **kwargs):
#         super().__init__()
#         self.maxpool2d = nn.MaxPool2d(kernel_size)
#
#     def forward(self, x):
#         origin_shape = x.shape
#         if len(origin_shape) > 4:
#             x = x.reshape(np.prod(origin_shape[0:-3]), *origin_shape[-3:])
#
#         x = self.maxpool2d(x)
#
#         if len(origin_shape) > 4:
#             x = x.reshape(*origin_shape[0:-3], *x.shape[-3:])
#         return x
#

class BaseLinearModule(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias=True,
                 node=LIFNode,
                 *args,
                 **kwargs):
        super().__init__()
        if node is None:
            raise TypeError

        self.groups = kwargs['groups'] if 'groups' in kwargs else 1
        if self.groups == 1:
            self.fc = nn.Linear(in_features=in_features,
                                out_features=out_features, bias=bias)
        else:
            self.fc = nn.ModuleList()
            for i in range(self.groups):
                self.fc.append(nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias
                ))
        self.node = partial(node, **kwargs)()

    def forward(self, x):
        if self.groups == 1:  # (t b) c
            outputs = self.fc(x)

        else:  # b (c t)
            x = rearrange(x, 'b (c t) -> t b c', t=self.groups)
            outputs = []
            for i in range(self.groups):
                outputs.append(self.fc[i](x[i]))
            outputs = torch.stack(outputs)  # t b c
            outputs = rearrange(outputs, 't b c -> b (c t)')

        return self.node(outputs)


class BaseConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 bias=False,
                 node=LIFNode,
                 conv_fun=nn.Conv2d,
                 **kwargs):
        super().__init__()
        if node is None:
            raise TypeError

        self.groups = kwargs['groups'] if 'groups' in kwargs else 1
        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False
        # self.conv = nn.Conv2d(
        #     in_channels=in_channels * self.groups,
        #     out_channels=out_channels * self.groups,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     stride=stride,
        #     bias=bias
        # )
        self.conv = conv_fun(
            in_channels=in_channels * self.groups,
            out_channels=out_channels * self.groups,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        # self.conv = DeformConvPack(in_channels=in_channels,
        #                            out_channels=out_channels,
        #                            kernel_size=kernel_size,
        #                            padding=padding,
        #                            stride=stride,
        #                            bias=bias)

        # self.ca = ChannelAttention(out_channels, out_channels)
        # self.sa = SpatialAttention()

        if self.n_preact:
            self.preact = nn.Sequential(
                LearnableBias(out_channels),
                nn.PReLU(out_channels, init=.25),
                # PeLU(out_channels),
                LearnableBias(out_channels),
            )
        else:
            self.preact = nn.Identity()
            # self.preact = Maxout(out_channels)
        # print(self.n_preact)

        self.bn = nn.BatchNorm2d(out_channels * self.groups)
        # self.bn = nn.InstanceNorm2d(out_channels * self.groups)

        self.node = partial(node, channel=out_channels, **kwargs)()
        # self.node = node()

        self.activation = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = 2. * x * self.ca(x)
        # x = 2. * x * self.sa(x)
        x = self.preact(x)
        x = self.node(x)
        # x = self.node(F.relu(x))
        return x


class MultiConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 bias=False,
                 node=LIFNode,
                 conv_fun=nn.Conv2d,
                 **kwargs):
        super().__init__()
        if node is None:
            raise TypeError

        self.is_skip = (in_channels == out_channels)

        self.groups = kwargs['groups'] if 'groups' in kwargs else 1
        self.n_preact = kwargs['n_preact'] if 'n_preact' in kwargs else False
        # self.conv = nn.Conv2d(
        #     in_channels=in_channels * self.groups,
        #     out_channels=out_channels * self.groups,
        #     kernel_size=kernel_size,
        #     padding=padding,
        #     stride=stride,
        #     bias=bias
        # )
        self.conv5x5 = conv_fun(
            in_channels=in_channels * self.groups,
            out_channels=out_channels * self.groups,
            kernel_size=(5, 5),
            padding=2,
            stride=stride,
            bias=bias,
        )
        self.conv3x3_1 = conv_fun(
            in_channels=in_channels * self.groups,
            out_channels=out_channels * self.groups,
            kernel_size=(3, 3),
            padding=1,
            stride=stride,
            bias=False,
        )
        self.conv3x3_2 = conv_fun(
            in_channels=out_channels * self.groups,
            out_channels=out_channels * self.groups,
            kernel_size=(3, 3),
            padding=1,
            stride=stride,
            bias=bias,
        )
        self.conv1x1 = conv_fun(
            in_channels=in_channels * self.groups,
            out_channels=out_channels * self.groups,
            kernel_size=(1, 1),
            padding=0,
            stride=stride,
            bias=bias,
        )
        # self.conv = DeformConvPack(in_channels=in_channels,
        #                            out_channels=out_channels,
        #                            kernel_size=kernel_size,
        #                            padding=padding,
        #                            stride=stride,
        #                            bias=bias)

        # self.ca = ChannelAttention(out_channels, out_channels)
        # self.sa = SpatialAttention()

        if self.n_preact:
            self.preact = nn.Sequential(
                LearnableBias(out_channels),
                nn.PReLU(out_channels),
                # PeLU(out_channels),
                LearnableBias(out_channels),
            )
        else:
            self.preact = nn.Identity()
            # self.preact = Maxout(out_channels)
        # print(self.n_preact)

        self.bn5x5 = nn.BatchNorm2d(out_channels * self.groups)
        self.bn3x3_1 = nn.BatchNorm2d(out_channels * self.groups)
        self.bn3x3_2 = nn.BatchNorm2d(out_channels * self.groups)
        self.bn1x1 = nn.BatchNorm2d(out_channels * self.groups)

        # self.bn = nn.InstanceNorm2d(out_channels * self.groups)

        self.node = partial(node, channel=out_channels, **kwargs)()
        # self.node = node()

        self.activation = nn.Identity()

    def forward(self, x):
        x1 = self.conv5x5(x)
        x1 = self.bn5x5(x1)

        x2 = self.conv3x3_1(x)
        x2 = self.bn3x3_1(x2)
        x2 = self.conv3x3_2(x2)
        x2 = self.bn3x3_2(x2)

        x3 = self.conv1x1(x)
        x3 = self.bn1x1(x3)

        if self.is_skip:
            x = x1 + x2 + x3 + x
        else:
            x = x1 + x2 + x3
        # x = 2. * x * self.ca(x)
        # x = 2. * x * self.sa(x)
        x = self.preact(x)

        x = self.node(x)
        return x


class BaseModule(nn.Module, abc.ABC):
    def __init__(self,
                 step=10,
                 encode_type='direct',
                 layer_by_layer=False,
                 temporal_flatten=False,
                 adaptive_node=False,
                 temporal_output=False,
                 tet_loss=False,
                 online_training=False,
                 conv_type='normal',
                 *args,
                 **kwargs):
        super(BaseModule, self).__init__()
        self.step = step
        # print(kwargs['layer_by_layer'])
        self.layer_by_layer = layer_by_layer

        self.temporal_flatten = temporal_flatten
        self.adaptive_node = adaptive_node
        self.tet_loss = tet_loss
        self.temporal_output = temporal_output
        self.online_training = online_training

        conv_type = conv_type.lower()
        if conv_type == 'normal':
            self.conv_fun = nn.Conv2d
        elif conv_type == 'ws':
            self.conv_fun = WSConv2d
        else:
            raise NotImplementedError('{} Cannot be Recognized.'.format(conv_type))

        encode_step = self.step

        if temporal_flatten is True:
            self.init_channel_mul = self.step
            self.step = 1
        elif adaptive_node is True:
            self.init_channel_mul = 1
            self.step = 1
            encode_step = 1
        else:  # origin
            self.init_channel_mul = 1

        self.encoder = Encoder(
            step=encode_step,
            encode_type=encode_type,
            layer_by_layer=layer_by_layer,
            temporal_flatten=temporal_flatten,
            adaptive_node=adaptive_node,
            **kwargs
        )

        self.kwargs = kwargs
        self.warm_up = False

        self.fire_rate = []

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'n_reset'):
                mod.n_reset()

    def set_warm_up(self, flag):
        self.warm_up = flag
        for mod in self.modules():
            if hasattr(mod, 'set_n_warm_up'):
                mod.set_n_warm_up(flag)

    def set_threshold(self, thresh):
        for mod in self.modules():
            if hasattr(mod, 'set_n_threshold'):
                mod.set_n_threshold(thresh)

    def set_attr(self, attr, val):
        for mod in self.modules():
            # if isinstance(mod, BaseNode):
            if hasattr(mod, attr):
                setattr(mod, attr, val)
                # else:
                #     ValueError('{} do not has {}'.format(self, attr))

    def set_decay(self, decay):
        for mod in self.modules():
            if hasattr(mod, 'set_n_decay'):
                mod.set_n_decay(decay)

    def get_threshold(self):
        outputs = []
        for mod in self.modules():
            if isinstance(mod, BaseNode):
                thresh = (mod.get_thres())
                outputs.append(thresh)
        return outputs

    def get_decay(self):
        outputs = []
        for mod in self.modules():
            if hasattr(mod, 'decay'):
                outputs.append(float(mod.decay.detach().clone()))
        return outputs

    def set_requires_fp(self, flag):
        for mod in self.modules():
            if hasattr(mod, 'requires_fp'):
                mod.requires_fp = flag

    def get_fp(self, temporal_info=False):
        outputs = []
        for mod in self.modules():
            if isinstance(mod, BaseNode):
                if isinstance(mod.feature_map, float) or not mod.feature_map:
                    outputs.append(0.)
                else:
                    if temporal_info:
                        # outputs.append(mod.feature_map)
                        outputs.append(torch.stack(mod.feature_map))  # t, b, c, xx
                    else:
                        outputs.append(sum(mod.feature_map) / len(mod.feature_map))
        return outputs

    def get_fire_rate(self, requires_grad=False):
        outputs = []
        fp = self.get_attr('feature_map')
        for f in fp:
            if requires_grad is False:
                if len(f) == 0:
                    return torch.tensor([0.])
                outputs.append(((sum(f) / len(f)).detach() > 0.).float().mean())
            else:
                outputs.append(((sum(f) / len(f)) > 0.).float().mean())
        if len(outputs) == 0:
            return torch.tensor([0.])
        return torch.stack(outputs)

    def get_tot_spike(self):
        tot_spike = 0
        fp = self.get_attr('feature_map')
        for f in fp:
            if len(f) == 0:
                break
            tot_spike += sum(f).sum()
        return tot_spike

    def get_spike_info(self):
        spike_feature_list = self.get_fp(temporal_info=True)
        avg, var, spike = [], [], []
        avg_per_step = []
        for spike_feature in spike_feature_list:
            avg_list = []
            for spike_t in spike_feature:
                avg_list.append(float(spike_t.mean()))
            avg_per_step.append(avg_list)

            spike_feature = sum(spike_feature)
            num = np.prod(spike_feature.shape)
            avg.append(float(spike_feature.sum()))
            var.append(float(spike_feature.std()))
            lst = []
            for t in range(self.step + 1):
                lst.append(float((spike_feature == t).sum() / num))
                # lst.append(  # for mem storage
                #     float(torch.logical_and(spike_feature >= 2 * t / (self.step + 1) - 1,
                #                             spike_feature < 2 * (t + 1) / (self.step + 1) - 1).sum() / num))
            spike.append(lst)

        return avg, var, spike, avg_per_step

    def get_attr(self, attr):
        outputs = []
        for mod in self.modules():
            if hasattr(mod, attr):
                outputs.append(getattr(mod, attr))
        return outputs

    def get_noise_param(self):
        mu, var = [], []
        for mod in self.modules():
            if hasattr(mod, 'log_alpha'):
                alpha = torch.exp(mod.log_alpha)
                beta = torch.exp(mod.log_beta)
                mu.append(alpha / (alpha + beta))
                var.append(((alpha + 1) * alpha) / ((alpha + beta + 1) * (alpha + beta)))
        return mu, var

    def load_node_weight(self, ckpt, trainable=False):
        for mod in self.modules():
            if isinstance(mod, AdaptiveNode):
                mod.load_state_dict(ckpt['state_dict'], False)
                if not trainable:
                    for param in mod.parameters():
                        param.requires_grad = False

    @staticmethod
    def forward(self, inputs):
        pass
