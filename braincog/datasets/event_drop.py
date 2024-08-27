import math
import numpy as np
import torch
from torch.nn import functional as F


def drop_by_time(event):  # t, p, x, y
    step = event.shape[0]
    t_start = int(np.random.uniform(0, 1) * step)
    t_end = int(np.random.randint(t_start, step) / 10.0)
    event[t_start: t_end] = 0
    return event


def drop_by_area(event):
    size = event.shape
    xx = np.random.randint(0, size[2], (2,))
    yy = np.random.randint(0, size[3], (2,))
    xx.sort()
    yy.sort()
    event[:, :, xx[0]:xx[1], yy[0]:yy[1]] = 0
    return event


def random_drop(event):
    ratio = np.random.randint(1, 10) / 10.0
    mask = torch.rand_like(event)
    return torch.where(mask > ratio, event, 0)


def event_drop(events):

    option = np.random.randint(0, 4)  # 0: identity, 1: drop_by_time, 2: drop_by_area, 3: random_drop
    if option == 0:  # identity, do nothing
        return events
    elif option == 1:  # drop_by_time
        events = drop_by_time(events)
    elif option == 2:  # drop by area
        events = drop_by_area(events)
    elif option == 3:  # random drop
        events = random_drop(events)

    return events
