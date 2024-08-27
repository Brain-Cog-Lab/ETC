import torch
from torch.autograd import Variable

__all__ = ['criterions', 'gen_input_signal', 'visualization', 'drop_path']

from . import (
    criterions,
    gen_input_signal,
    visualization,
)


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(
            x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


