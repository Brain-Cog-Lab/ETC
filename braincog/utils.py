import os
import random
import math
import csv
import numpy as np
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from braincog.model_zoo.base_module import BaseModule, BaseConvModule

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def random_gradient(model: nn.Module, sigma: float):
    for param in model.parameters():
        if param.grad is None:
            continue
        noise = torch.randn_like(param) * sigma
        param.grad = param.grad + noise


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    """Compute the top1 and top5 accuracy
    """
    maxk = max(topk)
    batch_size = target.size(0)
    # Return the k largest elements of the given input tensor
    # along a given dimension -> N * k
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def mse(x, y):
    out = (x - y).pow(2).sum(-1, keepdim=True).mean()
    return out


def rand_ortho(shape, irange):
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return np.dot(U, np.dot(np.eye(U.shape[1], V.shape[0]), V))


def adjust_surrogate_coeff(epoch, tot_epochs):
    T_min, T_max = 1e-3, 1e1
    Kmin, Kmax = math.log(T_min) / math.log(10), math.log(T_max) / math.log(10)
    t = torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / tot_epochs * epoch)]).float().cuda()
    k = torch.tensor([1]).float().cuda()
    if k < 1:
        k = 1 / t
    return t, k


def save_feature_map(x, dir=''):
    for idx, layer in enumerate(x):
        layer = layer.cpu()
        for batch in range(layer.shape[0]):
            for channel in range(layer.shape[1]):
                fname = '{}_{}_{}_{}.jpg'.format(
                    idx, batch, channel, layer.shape[-1])
                fp = layer[batch, channel]
                plt.tight_layout()
                plt.axis('off')
                plt.imshow(fp, cmap='inferno')
                plt.savefig(os.path.join(dir, fname),
                            bbox_inches='tight', pad_inches=0)


def save_spike_info(fname, epoch, batch_idx, step, avg, var, spike, avg_per_step):

    if not os.path.exists(fname):
        f = open(fname, mode='w', encoding='utf8', newline='')
        writer = csv.writer(f)
        head = ['epoch', 'batch', 'layer', 'avg', 'var']
        head.extend(['st_{}'.format(i) for i in range(step + 1)])  # spike times
        head.extend(['as_{}'.format(i) for i in range(step)])  # avg spike per time
        writer.writerow(head)

    else:
        f = open(fname, mode='a', encoding='utf8', newline='')
        writer = csv.writer(f)

    for layer in range(len(avg)):
        lst = [epoch, batch_idx, layer, avg[layer], var[layer]]
        lst.extend(spike[layer])
        lst.extend(avg_per_step[layer])
        lst = [str(x) for x in lst]
        writer.writerow(lst)


def unpack_adaption_info(model: BaseModule):
    bias_x, weight, bias_y = [], [], []
    for mod in model.modules():
        if isinstance(mod, BaseConvModule):
            bias_x.append(mod.preact[0].bias.detach().cpu().flatten().tolist())
            weight.append(mod.preact[1].weight.detach().cpu().flatten().tolist())
            bias_y.append(mod.preact[2].bias.detach().cpu().flatten().tolist())
    return bias_x, weight, bias_y


def save_adaptation_info(fname, epoch, bias_x, weight, bias_y):
    if not os.path.exists(fname):
        f = open(fname, mode='w', encoding='utf8', newline='')
        writer = csv.writer(f)
        head = ['epoch', 'layer', 'channel', 'bias_x', 'weight', 'bias_y']
        writer.writerow(head)

    else:
        f = open(fname, mode='a', encoding='utf8', newline='')
        writer = csv.writer(f)

    for layer in range(len(bias_x)):
        for idx in range(len(bias_x[layer])):
            lst = [epoch, layer, idx, bias_x[layer][idx], weight[layer][idx], bias_y[layer][idx]]
            writer.writerow(lst)

