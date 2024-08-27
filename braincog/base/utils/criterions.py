# Thanks to rwightman's timm package
# github.com:rwightman/pytorch-image-models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from braincog.model_zoo.base_module import BaseModule


def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = torch.nn.LogSoftmax(dim=1)
    if size_average:
        return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    else:
        return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def calc_critical_loss(model: BaseModule):
    # Control Mu in noise
    # mu, _ = model.get_noise_param()
    # mu = torch.stack(mu)
    # fire_rate = -model.get_fire_rate(requires_grad=True)
    # return cross_entropy(mu.unsqueeze(0), torch.softmax(fire_rate.unsqueeze(0), dim=1))

    # BP for fire_rate
    # fire_rate = model.get_fire_rate(requires_grad=True)
    # return F.mse_loss(fire_rate, fire_rate.mean().detach())
    # return F.mse_loss(fire_rate, torch.tensor(0.12, device=fire_rate.device)), fire_rate.tolist()

    # Direct Change threshold
    # threshold = model.get_attr('threshold')
    # fire_rate = model.get_fire_rate(requires_grad=False)
    # fire_rate = fire_rate - fire_rate.mean()
    # for i in range(len(threshold)):
    #     threshold[i].data = threshold[i] + 0.005 * fire_rate[i]
    # return fire_rate.detach().std()

    # change the distribution of mem, # TODO: only consider last step
    # mem_list = model.get_attr('mem')  # [node1.mem, node2.mem, ...]
    # loss = torch.tensor(0.).cuda()
    # for mem in mem_list:
    #     mem_upper = torch.where(mem > 0., mem, torch.zeros_like(mem))
    #     mem_lower = torch.where(mem < 0., mem, torch.zeros_like(mem))
    #     loss += F.mse_loss(mem_upper, torch.tensor(1.).cuda()) + F.mse_loss(mem_lower, torch.tensor(-1.).cuda())

    mem_list = model.get_attr('saved_mem')
    loss = 0.
    for mem in mem_list:
        # print(mem.mean())
        # print(float(mem.mean()), float(mem.std()), float(mem.max()), float(mem.min()))
        loss += F.mse_loss(mem.mean(), torch.tensor(0.).cuda()) - mem.std()
    return loss / len(mem_list)


class CutMixCrossEntropyLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average

    def forward(self, input, target):
        if len(target.shape) == 1:
            target = torch.nn.functional.one_hot(target, num_classes=input.size(-1))
            target = target.float().cuda()
        return cross_entropy(input, target, self.size_average)


class LabelSmoothingBCEWithLogitsLoss(nn.Module):

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingBCEWithLogitsLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        target = torch.eye(x.shape[-1], device=x.device)[target]
        nll = torch.ones_like(x) / x.shape[-1]
        return self.BCELoss(x, target) * self.confidence + self.BCELoss(x, nll) * self.smoothing


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def _compute_losses(self, x, target):
        log_prob = F.log_softmax(x, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss

    def forward(self, x, target):
        return self._compute_losses(x, target).mean()


class SoftCrossEntropy(torch.nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, inputs, targets, temperature=1.):
        log_likelihood = -F.log_softmax(inputs / temperature, dim=1)
        likelihood = F.softmax(targets / temperature, dim=1)
        sample_num, class_num = targets.shape
        loss = torch.sum(torch.mul(log_likelihood, likelihood)) / sample_num
        return loss


class UnilateralMse(torch.nn.Module):
    def __init__(self, thresh=1.):
        super(UnilateralMse, self).__init__()
        self.thresh = thresh
        self.loss = torch.nn.MSELoss()

    def forward(self, x, target):
        # x = nn.functional.softmax(x, dim=1)
        torch.clip(x, max=self.thresh)
        if x.shape == target.shape:
            return self.loss(x, target)
        return self.loss(x, torch.zeros_like(x).scatter_(1, target.view(-1, 1), self.thresh))


class MyBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super(MyBCEWithLogitsLoss, self).__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, target):
        return self.loss(x, torch.zeros(x.shape, device=x.device).scatter_(1, target.view(-1, 1), 1.))


class MixLoss(torch.nn.Module):
    def __init__(self, ce_loss):
        super(MixLoss, self).__init__()
        self.ce = ce_loss
        self.mse = UnilateralMse(1.)

    def forward(self, x, target):
        return 0.1 * self.ce(x, target) + self.mse(x, target)


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        print(gamma)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class TetLoss(torch.nn.Module):
    def __init__(self, loss_fn):
        super(TetLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, x, target):
        loss = 0.
        for logit in x:
            loss += self.loss_fn(logit, target)

        return loss / x.shape[0]


class ETC(nn.Module):
    def __init__(self, temp=4):
        super(ETC, self).__init__()
        self.T = temp

    def forward(self, outputs):
        loss = 0.
        count = 0
        for i in range(len(outputs)):
            for j in range(len(outputs)):
                if i != j:
                    p_s = F.log_softmax(outputs[i] / self.T, dim=1)
                    p_t = F.softmax(outputs[j] / self.T, dim=1)
                    loss += F.kl_div(p_s, p_t.detach(), size_average=False) * (self.T ** 2) / outputs[0].shape[0]
                    count += 1
        return loss / count
