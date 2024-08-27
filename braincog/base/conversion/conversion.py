from cgitb import Hook
from turtle import forward
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from copy import deepcopy


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class ImageNetPolicy(object):
    """ Randomly choose one of the best 24 Sub-policies on ImageNet.

        Example:
        >>> policy = ImageNetPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     ImageNetPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"


class CIFAR10Policy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.

        Example:
        >>> policy = CIFAR10Policy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     CIFAR10Policy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"


class SVHNPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on SVHN.

        Example:
        >>> policy = SVHNPolicy()
        >>> transformed = policy(image)

        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     SVHNPolicy(),
        >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.9, "shearX", 4, 0.2, "invert", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.7, "invert", 5, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.6, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 3, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "equalize", 1, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.8, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "shearY", 8, 0.4, "invert", 5, fillcolor),
            SubPolicy(0.9, "shearY", 5, 0.2, "solarize", 6, fillcolor),
            SubPolicy(0.9, "invert", 6, 0.8, "autocontrast", 1, fillcolor),
            SubPolicy(0.6, "equalize", 3, 0.9, "rotate", 3, fillcolor),

            SubPolicy(0.9, "shearX", 4, 0.3, "solarize", 3, fillcolor),
            SubPolicy(0.8, "shearY", 8, 0.7, "invert", 4, fillcolor),
            SubPolicy(0.9, "equalize", 5, 0.6, "translateY", 6, fillcolor),
            SubPolicy(0.9, "invert", 4, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.3, "contrast", 3, 0.8, "rotate", 4, fillcolor),

            SubPolicy(0.8, "invert", 5, 0.0, "translateY", 2, fillcolor),
            SubPolicy(0.7, "shearY", 6, 0.4, "solarize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 0.8, "rotate", 4, fillcolor),
            SubPolicy(0.3, "shearY", 7, 0.9, "translateX", 3, fillcolor),
            SubPolicy(0.1, "shearX", 6, 0.6, "invert", 5, fillcolor),

            SubPolicy(0.7, "solarize", 2, 0.6, "translateY", 7, fillcolor),
            SubPolicy(0.8, "shearY", 4, 0.8, "invert", 8, fillcolor),
            SubPolicy(0.7, "shearX", 9, 0.8, "translateY", 3, fillcolor),
            SubPolicy(0.8, "shearY", 5, 0.7, "autocontrast", 3, fillcolor),
            SubPolicy(0.7, "shearX", 2, 0.1, "invert", 5, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment SVHN Policy"


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude *
                                         random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude *
                                         random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude *
                                         img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude *
                                         img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        # self.name = "{}_{:.2f}_and_{}_{:.2f}".format(
        #     operation1, ranges[operation1][magnitude_idx1],
        #     operation2, ranges[operation2][magnitude_idx2])
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


def evaluate_accuracy(data_iter, net, device=None, only_onebatch=False, ind=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(data_iter)):
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]

            if only_onebatch: break
            if i == ind: break
    return acc_sum / n


def fuseConvBN(m):
    children = list(m.named_children())
    c, cn = None, None

    for i, (name, child) in enumerate(children):
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = torch.nn.Identity()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuseConvBN(child)
    return m


def fuse(conv, bn):
    w = conv.weight
    mean, var_sqrt, beta, gamma = bn.running_mean, torch.sqrt(bn.running_var + bn.eps), bn.weight, bn.bias
    b = conv.bias if conv.bias is not None else mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean) / var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def load_imagenet(root='/data/raid/floyed/ILSVRC2012', batch_size=128, train_batch=None):
    '''
    load imagenet 2012
    we use images in train/ for training, and use images in val/ for testing
    https://github.com/pytorch/examples/tree/master/imagenet
    '''
    IMAGENET_PATH = root
    traindir = os.path.join(IMAGENET_PATH, 'train')
    valdir = os.path.join(IMAGENET_PATH, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize]))

    batch = batch_size if train_batch is None else train_batch
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch, shuffle=False,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, val_loader, train_dataset, val_dataset


def seed_all(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class HookScale(nn.Module):
    def __init__(self, p=0.9995, channelnorm=False, gamma=1):
        super().__init__()
        if channelnorm:
            self.register_buffer('scale', torch.tensor(1.0))
        else:
            self.register_buffer('scale', torch.tensor(1.0))
        
        self.p = p
        self.channelnorm = channelnorm
        self.gamma = gamma

    def forward(self, x):
        x = x.detach()

        with torch.no_grad():
            if self.gamma > 1:
                x[x>self.gamma] = self.gamma
            if len(x.shape) == 4 and self.channelnorm:
                num_channel = x.shape[1]
                self.scale = torch.quantile(x.permute(1, 0, 2, 3).reshape(num_channel, -1), self.p, dim=1, interpolation='lower') + 1e-10
            else:
                sort, _ = torch.sort(x.view(-1))
                self.scale = sort[int(sort.shape[0] * self.p) - 1]
        return x


class Converter(nn.Module):
    def __init__(self, dataloader, device=None, p=0.9995, channelnorm=False, lateral_inhi=True, gamma=1, spicalib=False, monitor=False, smode=True, allowance=256):
        super().__init__()
        self.dataloader = dataloader
        self.device = device
        self.p = p
        self.channelnorm = channelnorm
        self.lateral_inhi = lateral_inhi
        self.monitor = monitor
        self.smode =  smode
        self.spicalib = spicalib
        self.gamma = gamma
        self.allowance = allowance
        
    def forward(self, model):
        model.eval()
        print('register hook scale...')
        self.register_hook(model, self.p, self.channelnorm, self.gamma).to(self.device)
        data = iter(self.dataloader).next()[0].to(self.device)
        # print(data[0][0])
        model(data)
        torch.cuda.empty_cache()
        print('finished!')
        model = self.replace_for_spike(model, smode=self.smode, gamma=self.gamma, lateral_inhi=self.lateral_inhi, spicalib=self.spicalib, monitor=self.monitor, allowance=self.allowance)
        return model

    @staticmethod
    def register_hook(model, p=0.99, channelnorm=False, gamma=1):
        children = list(model.named_children())
        for i, (name, child) in enumerate(children):
            if isinstance(child, nn.ReLU):
                model._modules[name] = nn.Sequential(nn.ReLU(), HookScale(p, channelnorm, gamma)) 
            elif isinstance(child, nn.Conv2d):
                model._modules[name] = nn.Sequential(child, HookScale(p, channelnorm, gamma)) 
            elif isinstance(child, nn.BatchNorm2d):
                model._modules[name] = nn.Sequential(child, HookScale(p, channelnorm, gamma)) 
            elif isinstance(child, nn.MaxPool2d):
                model._modules[name] = nn.Sequential(child, HookScale(p, channelnorm, gamma)) 
            else:
                Converter.register_hook(child, p, channelnorm, gamma)
        return model

    @staticmethod
    def replace_for_spike(model, smode=True, gamma=1, lateral_inhi=True, spicalib=False, monitor=False, allowance=256):
        children = list(model.named_children())
        for i, (name, child) in enumerate(children):
            if isinstance(child, nn.Sequential) and len(child) == 2 and isinstance(child[0], nn.ReLU) and isinstance(child[1], HookScale):
                if isinstance(allowance, list):
                    allow = allowance[0]
                    del allowance[0]
                else:
                    allow = allowance
                # print(child[1].scale)
                model._modules[name] = nn.Sequential(
                    Scale(1.0 / child[1].scale),
                    SNode(smode=smode, gamma=gamma, spicalib=spicalib, monitor=monitor, allowance=allow),
                    Scale(child[1].scale)) 
            elif isinstance(child, nn.Sequential) and len(child) == 2 and isinstance(child[0], nn.Conv2d) and isinstance(child[1], HookScale):
                model._modules[name] = child[0]
            elif isinstance(child, nn.Sequential) and len(child) == 2 and isinstance(child[0], nn.BatchNorm2d) and isinstance(child[1], HookScale):
                model._modules[name] = child[0]
            elif isinstance(child, nn.Sequential) and len(child) == 2 and isinstance(child[0], nn.MaxPool2d) and isinstance(child[1], HookScale):
                model._modules[name] = SMaxPool(child[0], smode=smode, lateral_inhi=lateral_inhi, monitor=monitor)
            else:
                Converter.replace_for_spike(child, smode, gamma, lateral_inhi, spicalib, monitor, allowance)
        return model


def clean_mem_spike(m):
    children = list(m.named_children())
    for name, child in children:
        if isinstance(child, SNode):
            child.mem = 0
            child.spike = 0
            child.sum = 0
            child.rmem = []
            child.rspike = []
            child.rsum = []

            child.sumspike = 0
            child.summem = 0
            child.t = 0
            
        elif isinstance(child, SMaxPool):
            child.input = 0
            child.sum = 0
            child.sumspike = 0
        else:
            clean_mem_spike(child)


class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.register_buffer('scale', scale)

    def forward(self, x):
        if len(self.scale.shape) == 1:
            return self.scale.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        else:
            return self.scale * x


class SNode(nn.Module):
    def __init__(self, smode=True, gamma=5, spicalib=False, monitor=False, allowance=256):
        super(SNode, self).__init__()
        self.smode = smode
        self.spicalib = spicalib
        self.monitor = monitor
        self.threshold = 1.0
        self.opration = nn.ReLU(True)
        self.gamma = gamma

        self.mem = 0
        self.spike = 0
        self.sum = 0
        
        self.rmem = []
        self.rspike = []
        self.rsum = []

        self.summem = 0
        self.sumspike = 0
        
        self.last_spike = 0
        self.avg_time = 0
        self.num_spike = 0
        self.t = 0

        self.allowance = allowance

    def forward(self, x):
        if not self.smode:
            if self.monitor: self.sum += x.cpu().detach()
            out = self.opration(x)
        else:
            self.mem = self.mem + x
            self.spike = (self.mem / self.threshold).floor().clamp(min=0, max=self.gamma)
            self.mem = self.mem - self.spike

            out = self.spike
            if self.monitor:
                self.summem += x
                self.rmem.append(self.summem.cpu().detach())
                self.rspike.append(self.sumspike.cpu().detach())
        return out


class SMaxPool(nn.Module):
    def __init__(self, child, smode=True, lateral_inhi=False, monitor=False):
        super(SMaxPool, self).__init__()
        self.smode = smode
        self.lateral_inhi = lateral_inhi
        self.monitor = monitor
        self.opration = child
        self.sum = 0
        self.input = 0
        self.sumspike = 0

    def forward(self, x):
        if self.monitor: self.input += x.detach().cpu()
        if not self.smode:
            out = self.opration(x)
        elif not self.lateral_inhi:
            self.sumspike += x
            single = self.opration(self.sumspike * 1000)
            sum_plus_spike = self.opration(x + self.sumspike * 1000)
            out = sum_plus_spike - single
        else:
            self.sumspike += x
            out = self.opration(self.sumspike)
            self.sumspike -= F.interpolate(out, scale_factor=2, mode='nearest')
        if self.monitor: self.sum += out.cpu().detach()
        return out
