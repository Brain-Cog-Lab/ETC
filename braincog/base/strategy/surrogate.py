import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Parameter

def heaviside(x):
    return (x >= 0.).to(x.dtype)


class SurrogateFunctionBase(nn.Module):
    """
    Surrogate Function 的基类
    :param alpha: 为一些能够调控函数形状的代理函数提供参数.
    :param requires_grad: 参数 ``alpha`` 是否需要计算梯度, 默认为 ``False``
    """

    def __init__(self, alpha, requires_grad=True):
        super().__init__()
        self.alpha = nn.Parameter(
            torch.tensor(alpha, dtype=torch.float),
            requires_grad=requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        """
        :param x: 膜电位的输入
        :param alpha: 控制代理梯度形状的变量, 可以为 ``NoneType``
        :return: 激发之后的spike, 取值为 ``[0, 1]``
        """
        raise NotImplementedError

    def forward(self, x, alpha=None):
        """
        :param x: 膜电位输入
        :return: 激发之后的spike
        """
        return self.act_fun(x, self.alpha)


'''
    sigmoid surrogate function.
'''


class sigmoid(torch.autograd.Function):
    """
    使用 sigmoid 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\mathrm{sigmoid}(\\alpha x) = \\frac{1}{1+e^{-\\alpha x}}
    反向传播的函数为:

    .. math::
            g'(x) = \\alpha * (1 - \\mathrm{sigmoid} (\\alpha x)) \\mathrm{sigmoid} (\\alpha x)
    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            s_x = torch.sigmoid(ctx.alpha * ctx.saved_tensors[0])
            grad_x = grad_output * s_x * (1 - s_x) * ctx.alpha
        return grad_x, None


class SigmoidGrad(SurrogateFunctionBase):
    def __init__(self, alpha=1., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return sigmoid.apply(x, alpha)


'''
    atan surrogate function.
'''


class atan(torch.autograd.Function):
    """
    使用 Atan 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\frac{1}{\\pi} \\arctan(\\frac{\\pi}{2}\\alpha x) + \\frac{1}{2}
    反向传播的函数为:

    .. math::
            g'(x) = \\frac{\\alpha}{2(1 + (\\frac{\\pi}{2}\\alpha x)^2)}
    """

    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.save_for_backward(inputs, alpha)
        return inputs.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        grad_alpha = None

        shared_c = grad_output / \
                   (1 + (ctx.saved_tensors[1] * math.pi /
                         2 * ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_x = ctx.saved_tensors[1] / 2 * shared_c
        if ctx.needs_input_grad[1]:
            grad_alpha = (ctx.saved_tensors[0] / 2 * shared_c).sum()

        return grad_x, grad_alpha


class AtanGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=True):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return atan.apply(x, alpha)


'''
    gate surrogate fucntion.
'''


class gate(torch.autograd.Function):
    """
    使用 gate 作为代理梯度函数
    对应的原函数为:

    .. math::
            g(x) = \\mathrm{NonzeroSign}(x) \\log (|\\alpha x| + 1)
    反向传播的函数为:

    .. math::
            g'(x) = \\frac{\\alpha}{1 + |\\alpha x|} = \\frac{1}{\\frac{1}{\\alpha} + |x|}
    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            grad_x = torch.where(x.abs() < 1. / alpha, torch.ones_like(x), torch.zeros_like(x))
            ctx.save_for_backward(grad_x)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None


class GateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return gate.apply(x, alpha)


'''
    gatquadratic_gate surrogate function.
'''


class quadratic_gate(torch.autograd.Function):
    """
    使用 quadratic_gate 作为代理梯度函数
    对应的原函数为:

    .. math::
        g(x) =
        \\begin{cases}
        0, & x < -\\frac{1}{\\alpha} \\\\
        -\\frac{1}{2}\\alpha^2|x|x + \\alpha x + \\frac{1}{2}, & |x| \\leq \\frac{1}{\\alpha}  \\\\
        1, & x > \\frac{1}{\\alpha} \\\\
        \\end{cases}
    反向传播的函数为:

    .. math::
        g'(x) =
        \\begin{cases}
        0, & |x| > \\frac{1}{\\alpha} \\\\
        -\\alpha^2|x|+\\alpha, & |x| \\leq \\frac{1}{\\alpha}
        \\end{cases}
    """

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            mask_zero = (x.abs() > 1 / alpha)
            grad_x = -alpha * alpha * x.abs() + alpha
            grad_x.masked_fill_(mask_zero, 0)
            ctx.save_for_backward(grad_x)
        return x.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None


class QGateGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return quadratic_gate.apply(x, alpha)


'''
    mollifier
'''


class mollifier(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            mask_zero = (x.abs() >= 1 / alpha)
            grad_x = alpha * alpha * torch.exp(1 / ((alpha * x.abs()) ** 2 - 1))
            grad_x.masked_fill_(mask_zero, 0)
            ctx.save_for_backward(grad_x)
        return x.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * ctx.saved_tensors[0]
        return grad_x, None


class MollifierGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return mollifier.apply(x, alpha)


'''
    test SG
'''


class TestGrad(nn.Module):
    def __init__(self, alpha=2., requires_grad=False):
        super(TestGrad, self).__init__()

    def forward(self, x):
        out_forward = torch.where(x > .5, torch.ones_like(x), torch.zeros_like(x))
        out_forward += torch.where(x < -.5, -torch.ones_like(x), torch.zeros_like(x))

        mask1 = x < 0.
        mask2 = x < .5
        mask3 = x < 1.
        mask4 = x < 1.5
        mask5 = x < 2.

        out1 = -1 * mask1 + (1 - mask1.float()) * ((x + 1.) ** 2 - .75)
        out2 = out1 * mask2.float() + (1 - mask2.float()) * (-x ** 2)
        out3 = out2 * mask3.float() + (1 - mask3.float()) * (x ** 2)
        out4 = out3 * mask4.float() + (1 - mask4.float()) * (-(x - 1.) ** 2 + .75)
        out = out4 * mask5.float() + (1 - mask5.float()) * 1.

        out = out_forward + out - out.detach()

        return out


class TestGradv2(nn.Module):
    def __init__(self, alpha=2., requires_grad=False):
        super(TestGradv2, self).__init__()

    def forward(self, x):
        out_forward = heaviside(x)

        mask1 = x < -.5
        mask2 = x < 0.
        mask3 = x < .5

        out1 = (1 - mask1.float()) * (2 * (x + .5) * (x + .5))
        out2 = out1 * mask2.float() + (1 - mask2.float()) * (-2 * (x - .5) * (x - .5) + 1.)
        out = out2 * mask3.float()

        out = out_forward + out - out.detach()

        return out


class relu_like(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x, alpha)
        return heaviside(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_alpha = None, None
        x, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * x.gt(0.).float() * alpha
        if ctx.needs_input_grad[1]:
            grad_alpha = (grad_output * F.relu(x)).sum()
        return grad_x, grad_alpha


class ReLUGrad(SurrogateFunctionBase):
    """
    使用ReLU作为代替梯度函数, 主要用为相同结构的ANN的测试
    """

    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return relu_like.apply(x, alpha)


'''
    Straight-Through (ST) Estimator
'''


class straight_through_estimator(torch.autograd.Function):
    """
    使用直通估计器作为代理梯度函数
    http://arxiv.org/abs/1308.3432
    """

    @staticmethod
    def forward(ctx, inputs):
        outputs = heaviside(inputs)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output
        return grad_x


class BinaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2)) * grad_output
        return grad_input, None, None


class BinaryActivation(nn.Module):
    '''
        learnable distance and center for activation
    '''

    def __init__(self, *args, **kwargs):
        super(BinaryActivation, self).__init__()

    def gradient_approx(self, x):
        '''
            gradient approximation
            (https://github.com/liuzechun/Bi-Real-net/blob/master/pytorch_implementation/BiReal18_34/birealnet.py)
        '''
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x * x + 2 * x) * (1 - mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x * x + 2 * x) * (1 - mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1 - mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

    def forward(self, x):
        x = self.gradient_approx(x)
        return x


# TODO:

class STGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return straight_through_estimator.apply(x)


class trunc_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        # ctx.save_for_backward(inputs)
        return inputs.trunc()

    @staticmethod
    def backward(ctx, grad_output):
        # inputs, = ctx.saved_tensors
        grad_inputs = grad_output.clone()
        return grad_inputs


class RoundGrad(nn.Module):
    def __init__(self, **kwargs):
        super(RoundGrad, self).__init__()
        self.act = nn.Hardtanh(-.5, 4.5)

    def forward(self, x):
        x = self.act(x)
        return x.ceil() + x - x.detach()


class ExecInhGrad(nn.Module):
    def __init__(self, **kwargs):
        super(ExecInhGrad, self).__init__()
        self.act = nn.Hardtanh(-1.5, .5)

    def forward(self, x):
        x = self.act(x)
        return x.ceil() + x - x.detach()


class stdp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        outputs = inputs.gt(0.).float()
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs, = ctx.saved_tensors
        return inputs * grad_output


class STDPGrad(SurrogateFunctionBase):
    def __init__(self, alpha=2., requires_grad=False):
        super().__init__(alpha, requires_grad)

    @staticmethod
    def act_fun(x, alpha):
        return stdp.apply(x)


# class FloorGrid(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, inputs):
#         # ctx.save_for_backward(inputs)
#         return inputs.floor()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         inputs, = ctx.saved_tensors
#         # grad_inputs = grad_output.clone()
#         return inputs.gt(1.).float()
#
#

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#
#     # f = TestGradv2()
#     # x = torch.arange(-2, 2, 0.001)
#     # y = f(x)
#     # plt.plot(x, y)
#     # plt.show()
#
#     x = torch.arange(-2, 2, 0.01, requires_grad=True)
#     f = AdaSurGrad(alpha=2., hidden_dim=8)
#     # f = NoisyGateGrad(alpha=2.)
#     y = f(x)
#     plt.plot(x.detach().numpy(), y.detach().numpy())
#     plt.show()
#     y.sum().backward()
#     plt.plot(x.detach().numpy(), x.grad.numpy())
#     plt.show()
