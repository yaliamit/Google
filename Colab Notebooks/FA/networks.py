from torch import nn
from layers import FALinear
from utils import hinge_loss
import torch.optim as optim

def update_pars(pars,net,fix, i):

    if pars.layerwise:
      params=net.parameters()
    else:
      params=list(fix.parameters())+list(net.parameters())
    if pars.OPT=='SGD':
      pars.optimizer = optim.SGD(params, lr=pars.LR[i])
    else:
      pars.optimizer = optim.Adam(params)

def get_net_c(i, pars, fix, layer_pars):
    filter_width = 5
    filter_padding = 2
    if i == 3:
        filter_width = 3
        filter_padding = 1
    layer = nn.Sequential(
        nn.Conv2d(layer_pars.NCOLD, layer_pars.NC, filter_width, padding=filter_padding),
        nn.Hardtanh(min_val=0, max_val=pars.MX),
        nn.Dropout(.5),
        nn.MaxPool2d(2, stride=2)
    )

    layer_pars.HW //= 2
    layer_pars.NCOLD = layer_pars.NC
    layer_pars.NC *= 4

    if pars.fa:
        fa = FALinear(layer_pars.NCOLD * 2 * 2, 10)
    else:
        fa = nn.Linear(layer_pars.NCOLD * 2 * 2, 10)

    aux = nn.Sequential(
        nn.AvgPool2d(int(layer_pars.HW / 2), int(layer_pars.HW / 2)),
        nn.Flatten(),
        fa
    )

    net = nn.Sequential()
    net.add_module('layer', layer)
    net.add_module('aux', aux)

    update_pars(pars,net,fix, i)

    return net, fix, layer


def get_net_b(i, pars, fix, layer_pars):
    filter_width = 5
    filter_padding = 2
    if i == 3:
        filter_width = 3
        filter_padding = 1
    layer = nn.Sequential(
        nn.Conv2d(layer_pars.NCOLD, layer_pars.NC, filter_width, padding=filter_padding),
        nn.Hardtanh(min_val=0, max_val=pars.MX),
        nn.Dropout(.5),
        nn.MaxPool2d(2, stride=2)
    )

    layer_pars.HW //= 2
    layer_pars.NCOLD = layer_pars.NC
    layer_pars.NC *= 2

    if pars.fa:
        fa = FALinear(layer_pars.NCOLD * 2 * 2, 10)
    else:
        fa = nn.Linear(layer_pars.NCOLD * 2 * 2, 10)

    aux = nn.Sequential(
        nn.AvgPool2d(int(layer_pars.HW / 2), int(layer_pars.HW / 2)),
        nn.Flatten(),
        fa
    )

    net = nn.Sequential()
    net.add_module('layer', layer)
    net.add_module('aux', aux)

    update_pars(pars,net,fix, i)


    return net, fix, layer


def get_net_a(i, pars, fix, layer_pars):
    if (i == 2) or (i == 3):
        fix.add_module('max_pool%d' % i, nn.MaxPool2d(2, stride=2))
        layer_pars.HW /= 2

    layer = nn.Sequential(
        nn.Conv2d(layer_pars.NCOLD, layer_pars.NC, 3, padding=1),
        nn.Hardtanh(min_val=0, max_val=pars.MX)
    )
    layer_pars.NCOLD = layer_pars.NC

    if pars.fa:
        fa = FALinear(layer_pars.NC * 4, 10)
    else:
        fa = nn.Linear(layer_pars.NC * 4, 10)

    aux = nn.Sequential(
        nn.AvgPool2d(int(layer_pars.HW / 2), int(layer_pars.HW / 2)),
        nn.Flatten(),
        fa
    )

    net = nn.Sequential()
    net.add_module('layer', layer)
    net.add_module('aux', aux)

    update_pars(pars,net,fix,i)


    return net, fix, layer


def get_net_pre(pars):

    if pars.CR == 'CE':
        pars.criterion = nn.CrossEntropyLoss()
    else:
        pars.criterion = hinge_loss()

