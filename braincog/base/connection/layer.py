import warnings
import math
import numpy as np
import torch
from torch import nn
from torch import einsum
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from torch.nn import Parameter
from einops import rearrange


class Maxout(nn.Module):
    '''
        Nonlinear function
    '''

    def __init__(self, channel, neg_init=0.25, pos_init=1.0):
        super(Maxout, self).__init__()
        self.neg_scale = nn.Parameter(neg_init * torch.ones(channel))
        self.pos_scale = nn.Parameter(pos_init * torch.ones(channel))
        self.relu = nn.ReLU()

    def forward(self, x):
        # Maxout
        x = self.pos_scale.view(1, -1, 1, 1) * self.relu(x) - self.neg_scale.view(1, -1, 1, 1) * self.relu(-x)
        return x


class PeLU(nn.Module):
    def __init__(self, channels=1, device=None, dtype=None):
        super(PeLU, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty(channels, **factory_kwargs).fill_(np.log(3, )))
        # self.weight = Parameter(torch.empty(channels, **factory_kwargs).fill_(.25))

    def forward(self, x):
        # self.weight.data = torch.clamp(self.weight, 0., 1.)
        k = rearrange(self.weight, 'c -> 1 c 1 1')
        x_pos = torch.where(x > 0, x, torch.zeros_like(x))
        x_neg = torch.where(x < 0, k * (x.exp() - 1.) + (1. - k) * x, torch.zeros_like(x))
        return x_pos + x_neg


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, out_planes // ratio, (1, 1), bias=False),
                                nn.ReLU(),
                                nn.Conv2d(out_planes // ratio, out_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, stride=1):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, (kernel_size, kernel_size), padding=kernel_size // 2, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CustomLinear(nn.Module):

    def __init__(self, in_channels, out_channels, bias=True):
        super(CustomLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.weight = Parameter(torch.tensor([
        #     [1., .5, .25, .125],
        #     [0., 1., .5, .25],
        #     [0., 0., 1., .5],
        #     [0., 0., 0., 1.]
        # ]), requires_grad=True)
        self.weight = Parameter(torch.diag(torch.ones(self.in_channels)), requires_grad=True)
        # self.weight = Parameter(torch.randn(self.in_channels, self.in_channels))
        mask = torch.tril(torch.ones(self.in_channels, self.in_channels), diagonal=0)
        self.register_buffer('mask', mask)

        if bias:
            self.bias = Parameter(torch.zeros(out_channels), requires_grad=True)
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        weight = self.mask * self.weight
        return F.linear(inputs, weight, self.bias)


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, gain=True):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)

        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = 1.

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = self.gain * weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class VotingLayer(nn.Module):
    """
    用于SNNs的输出层, 几个神经元投票选出最终的类
    :param voter_num: 投票的神经元的数量, 例如 ``voter_num = 10``, 则表明会对这10个神经元取平均
    """

    def __init__(self, voter_num: int):
        super().__init__()
        self.voting = nn.AvgPool1d(voter_num, voter_num)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, voter_num * C]
        # ret.shape = [N, C]
        return self.voting(x.unsqueeze(1)).squeeze(1)


class WTALayer(nn.Module):
    def __init__(self, k=1):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C,W,H]
        # ret.shape = [N, C,W,H]
        pos = x * torch.rand(x.shape, device=x.device)
        if self.k > 1:
            x = x * (pos >= pos.topk(self.k, dim=1)[0][:, -1:]).float()
        else:
            x = x * (pos >= pos.max(1, True)[0]).float()

        return x


class NDropout(nn.Module):
    """
    与Drop功能相同, 但是会保证同一个样本不同时刻的mask相同.
    """

    def __init__(self, p):
        super(NDropout, self).__init__()
        self.p = p
        self.mask = None

    def n_reset(self):
        """
        重置, 能够生成新的mask
        :return:
        """
        self.mask = None

    def create_mask(self, x):
        """
        生成新的mask
        :param x: 输入Tensor, 生成与之形状相同的mask
        :return:
        """
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def forward(self, x):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return self.mask * x
        else:
            return x


class ThresholdDependentBatchNorm2d(_BatchNorm):
    """
    tdBN
    https://ojs.aaai.org/index.php/AAAI/article/view/17320
    """

    def __init__(self, num_features, alpha: float, threshold: float = .5, layer_by_layer: bool = True, affine: bool = True):
        self.alpha = alpha
        self.threshold = threshold

        super().__init__(num_features=num_features, affine=affine)

        assert layer_by_layer, \
            'tdBN may works in step-by-step mode, which will not take temporal dimension into batch norm'
        assert self.affine, 'ThresholdDependentBatchNorm needs to set `affine = True`!'

        torch.nn.init.constant_(self.weight, alpha * threshold)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input):
        output = super().forward(input)
        return output




class SMaxPool(nn.Module):
    """用于转换方法的最大池化层的常规替换
    选用具有最大脉冲发放率的神经元的脉冲通过，能够满足一般性最大池化层的需要

    Reference:
    https://arxiv.org/abs/1612.04052
    """

    def __init__(self, child):
        super(SMaxPool, self).__init__()
        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        self.sumspike += x
        single = self.opration(self.sumspike * 1000)
        sum_plus_spike = self.opration(x + self.sumspike * 1000)

        return sum_plus_spike - single

    def reset(self):
        self.sumspike = 0


class LIPool(nn.Module):
    r"""用于转换方法的最大池化层的精准替换
    LIPooling通过引入侧向抑制机制保证在转换后的SNN中输出的最大值与期望值相同。

    Reference:
    https://arxiv.org/abs/2204.13271
    """

    def __init__(self, child=None):
        super(LIPool, self).__init__()
        if child is None:
            raise NotImplementedError("child should be Pooling operation with torch.")

        self.opration = child
        self.sumspike = 0

    def forward(self, x):
        self.sumspike += x
        out = self.opration(self.sumspike)
        self.sumspike -= F.interpolate(out, scale_factor=2, mode='nearest')
        return out

    def reset(self):
        self.sumspike = 0


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
