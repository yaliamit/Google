"""
Functions from https://github.com/C16Mftang/biological-SSL/blob/main/utils.py for identical transformation.
"""
from functools import partial
import random
from typing import List

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
import pytorch_lightning as pl
from lightly.data.collate import BaseCollateFunction


def rgb_to_hsv(inp):
    device = inp.device
    inp = inp.transpose(1, 3)
    sh = inp.shape
    inp = inp.reshape(-1, 3)

    mx, inmx = torch.max(inp, dim=1)
    mn, inmc = torch.min(inp, dim=1)
    df = mx - mn
    h = torch.zeros(inp.shape[0], 1).to(device)
    # if False: #'xla' not in device.type:
    #     h.to(device)
    ii = [0, 1, 2]
    iid = [[1, 2], [2, 0], [0, 1]]
    shift = [360, 120, 240]

    for i, _id, s in zip(ii, iid, shift):
        logi = (df != 0) & (inmx == i)
        h[logi, 0] = \
            torch.remainder((60 * (inp[logi, _id[0]] - inp[logi, _id[1]]) / df[logi] + s), 360)

    s = torch.zeros(inp.shape[0], 1).to(device)
    s[mx != 0, 0] = (df[mx != 0] / mx[mx != 0]) * 100

    v = mx.reshape(inp.shape[0], 1) * 100

    output = torch.cat((h / 360., s / 100., v / 100.), dim=1)

    output = output.reshape(sh).transpose(1, 3)
    return output


def hsv_to_rgb(inp):
    device = inp.device
    inp = inp.transpose(1, 3)
    sh = inp.shape
    inp = inp.reshape(-1, 3)

    hh = inp[:, 0]
    hh = hh * 6
    ihh = torch.floor(hh).type(torch.int32)
    ff = (hh - ihh)[:, None]
    v = inp[:, 2][:, None]
    s = inp[:, 1][:, None]
    p = v * (1.0 - s)
    q = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)))

    output = torch.zeros_like(inp).to(device)
    
    output[ihh == 0, :] = torch.cat((v[ihh == 0], t[ihh == 0], p[ihh == 0]), dim=1)
    output[ihh == 1, :] = torch.cat((q[ihh == 1], v[ihh == 1], p[ihh == 1]), dim=1)
    output[ihh == 2, :] = torch.cat((p[ihh == 2], v[ihh == 2], t[ihh == 2]), dim=1)
    output[ihh == 3, :] = torch.cat((p[ihh == 3], q[ihh == 3], v[ihh == 3]), dim=1)
    output[ihh == 4, :] = torch.cat((t[ihh == 4], p[ihh == 4], v[ihh == 4]), dim=1)
    output[ihh == 5, :] = torch.cat((v[ihh == 5], p[ihh == 5], q[ihh == 5]), dim=1)

    output = output.reshape(sh)
    output = output.transpose(1, 3)
    return output


def deform_data(x_in, perturb, trans, s_factor, h_factor, embedd):
    device = x_in.device
    h = x_in.shape[2]
    w = x_in.shape[3]
    nn = x_in.shape[0]
    v = ((torch.rand(nn, 6) - .5) * perturb).to(device)
    rr = torch.zeros(nn, 6).to(device)
    if not embedd:
        ii = torch.randperm(nn)
        u = torch.zeros(nn, 6).to(device)
        u[ii[0:nn // 2]] = v[ii[0:nn // 2]]
    else:
        u = v
    # Amplify the shift part of the
    u[:, [2, 5]] *= 2
    rr[:, [0, 4]] = 1
    if trans == 'shift':
        u[:, [0, 1, 3, 4]] = 0
    elif trans == 'scale':
        u[:, [1, 3]] = 0
    elif 'rotate' in trans:
        u[:, [0, 1, 3, 4]] *= 1.5
        ang = u[:, 0]
        v = torch.zeros(nn, 6)
        v[:, 0] = torch.cos(ang)
        v[:, 1] = -torch.sin(ang)
        v[:, 4] = torch.cos(ang)
        v[:, 3] = torch.sin(ang)
        s = torch.ones(nn)
        if 'scale' in trans:
            s = torch.exp(u[:, 1])
        u[:, [0, 1, 3, 4]] = v[:, [0, 1, 3, 4]] * s.reshape(-1, 1).expand(nn, 4)
        rr[:, [0, 4]] = 0
    theta = (u + rr).view(-1, 2, 3)
    grid = F.affine_grid(theta, [nn, 1, h, w], align_corners=True)
    x_out = F.grid_sample(x_in, grid, padding_mode='zeros', align_corners=True)

    if x_in.shape[1] == 3 and s_factor > 0:
        v = torch.rand(nn, 2).to(device)
        vv = torch.pow(2, (v[:, 0] * s_factor - s_factor / 2)).reshape(nn, 1, 1)
        uu = ((v[:, 1] - .5) * h_factor).reshape(nn, 1, 1)
        x_out_hsv = rgb_to_hsv(x_out)
        x_out_hsv[:, 1, :, :] = torch.clamp(x_out_hsv[:, 1, :, :] * vv, 0., 1.)
        x_out_hsv[:, 0, :, :] = torch.remainder(x_out_hsv[:, 0, :, :] + uu, 1.)
        x_out = hsv_to_rgb(x_out_hsv)

    ii = torch.where(torch.bernoulli(torch.ones(nn) * .5) == 1)
    for i in ii:
        x_out[i] = x_out[i].flip(3)
    return x_out


def deform_gaze(x):
    n = x.shape[2]
    x1 = torch.cat((x[:, :, :n // 2, :n // 2], x[:, :, :n // 2, n // 2:]), dim=0)
    x2 = torch.cat((x[:, :, n // 2:, :n // 2], x[:, :, n // 2:, n // 2:]), dim=0)
    return x1, x2


def deform_gaze2(x, pars):
    bsz = x.size(0)
    patch_size = pars.patch_size
    x_unfold = F.unfold(x, kernel_size=patch_size, stride=patch_size // 2)  # (bsz, 256, 49)
    all_patches = x_unfold.permute(0, 2, 1).reshape(bsz * x_unfold.shape[-1], patch_size,
                                                    patch_size)  # (bsz*49, 16, 16)
    output = all_patches.unsqueeze(dim=1)  # bsz*49, 1, 16, 16
    return output


def random_rotate(image):
    if random.random() > 0.5:
        return tvf.rotate(image, angle=random.choice((0, 90, 180, 270)))
    return image


def get_scripted_transforms(s=1.0):
    tf = torch.nn.Sequential(
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(90),
        transforms.RandomApply(torch.nn.ModuleList([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        ]), p=0.8)
    )
    scripted_transforms = torch.jit.script(tf)
    return scripted_transforms

class BaseCollateFunctionCustomOp(pl.LightningModule):
    """Base class for other collate implementations. Takes a function that transforms a batch of tensor images.

    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.
    """

    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, batch: List[tuple]):
        """
            Args:
                batch:
                    A batch of tuples of images, labels, and filenames which
                    is automatically provided if the dataloader is built from
                    a LightlyDataset.

            Returns:
                A tuple of images, labels, and filenames. The images consist of
                two batches corresponding to the two transformations of the
                input images.

        """
        batch_size = len(batch)

        # list of transformed images
        to_tensor_fn = torchvision.transforms.ToTensor()
        tensor_images = [to_tensor_fn(batch[i % batch_size][0]).unsqueeze_(0)
                         for i in range(2 * batch_size)]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        transforms = (self.transform(torch.cat(tensor_images[:batch_size], 0)),
                      self.transform(torch.cat(tensor_images[batch_size:], 0)))

        return transforms, labels, fnames


def distortion3_collate_fn():
    return BaseCollateFunctionCustomOp(
        partial(deform_data, perturb=0.5, trans=['aff'], s_factor=4, h_factor=0.2, embedd=False))