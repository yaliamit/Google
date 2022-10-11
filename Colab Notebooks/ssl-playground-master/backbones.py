from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.batchnorm import get_norm_layer
from lightly.models.resnet import BasicBlock


class ResNetCifar(nn.Module):
    """
    ResNet for CIFAR-10, modified from lightly implementation of ResNet
    """

    def __init__(self,
                 block: nn.Module = BasicBlock,
                 layers: List[int] = [3, 3, 3],
                 num_classes: int = 10,
                 width: float = 1.,
                 num_splits: int = 0):
        super().__init__()
        self.in_planes = int(16 * width)

        self.base = int(16 * width)

        self.conv1 = nn.Conv2d(3,
                               self.base,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = get_norm_layer(self.base, num_splits)
        self.layer1 = self._make_layer(block, self.base, layers[0], stride=1, num_splits=num_splits)
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2, num_splits=num_splits)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2, num_splits=num_splits)
        self.linear = nn.Linear(self.base * 4 * block.expansion, num_classes)

    def _make_layer(self, block, planes, layers, stride, num_splits):
        strides = [stride] + [1] * (layers - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_splits))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNetCifarGenerator(name: str = 'resnet-20',
                         width: float = 1,
                         num_classes: int = 10,
                         num_splits: int = 0):
    model_params = {
        'resnet-20': {'block': BasicBlock, 'layers': [3, 3, 3]},
        'resnet-32': {'block': BasicBlock, 'layers': [5, 5, 5]},
        'resnet-44': {'block': BasicBlock, 'layers': [7, 7, 7]},
        'resnet-56': {'block': BasicBlock, 'layers': [9, 9, 9]},
        'resnet-110': {'block': BasicBlock, 'layers': [18, 18, 18]},
        'resnet-1202': {'block': BasicBlock, 'layers': [200, 200, 200]},
    }

    if name not in model_params.keys():
        raise ValueError('Illegal name: {%s}. \
        Try resnet-20, resnet-32, resnet-44, resnet-56, resnet-110, resnet-1202' % (name))

    return ResNetCifar(**model_params[name], width=width, num_classes=num_classes, num_splits=num_splits)


class Conv6(nn.Module):
    def __init__(self):
        super().__init__()

        layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, bias=False),
                               nn.Hardtanh())
        layer2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1, bias=False),
                               nn.MaxPool2d(2))
        layer3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, bias=False),
                               nn.Hardtanh())
        layer4 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1, bias=False),
                               nn.MaxPool2d(2))
        layer5 = nn.Sequential(nn.Conv2d(64, 512, 3, padding=1, bias=False),
                               nn.MaxPool2d(2))

        self.net = nn.Sequential(layer1, layer2, layer3, layer4, layer5)

    def forward(self, x):
        return self.net(x)


class Conv6Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        layer1 = nn.Sequential(nn.ConvTranspose2d(512, 64, 3, padding=1, output_padding=1, stride=2, bias=False),
                               nn.Hardtanh())
        layer2 = nn.Sequential(nn.ConvTranspose2d(64, 64, 3, padding=1, output_padding=1, stride=2, bias=False),
                               nn.Hardtanh())
        layer3 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, padding=1, bias=False),
                               nn.Hardtanh())
        layer4 = nn.Sequential(nn.ConvTranspose2d(32, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
                               nn.Hardtanh())
        layer5 = nn.Sequential(nn.ConvTranspose2d(32, 3, 3, padding=1, bias=False),
                               nn.Sigmoid())

        self.net = nn.Sequential(layer1, layer2, layer3, layer4, layer5)

    def forward(self, x):
        return self.net(x)
