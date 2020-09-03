from torch import nn
from layers import FALinear, FAConv2d
from utils import hinge_loss
import torch.optim as optim

def update_pars(pars,net,fix):

    lri=0
    if pars.layerwise:
      lri=pars.i
      params=net.parameters()
    else:
      params=list(fix.parameters())+list(net.parameters())
    if pars.OPT=='SGD':
      pars.optimizer = optim.SGD(params, lr=pars.LR[lri])
    else:
      pars.optimizer = optim.Adam(params)


def get_net_simp(pars, fix, layer_pars):

    filter_width = 5
    filter_padding = 2
    layer=nn.Sequential(
        FAConv2d(layer_pars.NCOLD, layer_pars.NC, filter_width, fa=pars.fa, device=pars.device, padding=filter_padding),
        nn.MaxPool2d(3, stride=2,padding=1),
        nn.Hardtanh(min_val=0, max_val=pars.MX),
        nn.Dropout(.8),
    )

    aux=nn.Sequential(
        nn.Flatten(),
        FALinear(16*16*layer_pars.NC, 500, fa=pars.fa),
        nn.Hardtanh(min_val=0, max_val=pars.MX),
        nn.Dropout(.8),
        FALinear(500, pars.num_class, fa=pars.fa),
    )


    net = nn.Sequential()
    net.add_module('layer', layer)
    net.add_module('aux', aux)

    update_pars(pars, net, fix)

    return net, fix, layer

def get_net_c(pars, fix, layer_pars):
    filter_width = 5
    filter_padding = 2
    if pars.i == 3:
        filter_width = 3
        filter_padding = 1
    layer = nn.Sequential(
        FAConv2d(layer_pars.NCOLD, layer_pars.NC, filter_width, fa=pars.fa, device=pars.device, padding=filter_padding),
        nn.Hardtanh(min_val=0, max_val=pars.MX),
        nn.Dropout(.5),
        nn.MaxPool2d(2, stride=2)
    )

    layer_pars.HW //= 2
    layer_pars.NCOLD = layer_pars.NC
    layer_pars.NC *= 4


    fal = FALinear(layer_pars.NCOLD * 2 * 2, pars.num_class, fa=pars.fa)


    aux = nn.Sequential(
        nn.AvgPool2d(int(layer_pars.HW / 2), int(layer_pars.HW / 2)),
        nn.Flatten(),
        fal
    )

    net = nn.Sequential()
    net.add_module('layer', layer)
    net.add_module('aux', aux)

    update_pars(pars,net,fix)

    return net, fix, layer


def get_net_b(pars, fix, layer_pars):
    filter_width = 5
    filter_padding = 2
    if pars.i == 3:
        filter_width = 3
        filter_padding = 1
    layer = nn.Sequential(
        FAConv2d(layer_pars.NCOLD, layer_pars.NC, filter_width, fa=pars.fa, device=pars.device, padding=filter_padding),
        nn.Hardtanh(min_val=0, max_val=pars.MX),
        nn.Dropout(.5),
        nn.MaxPool2d(2, stride=2)
    )

    layer_pars.HW //= 2
    layer_pars.NCOLD = layer_pars.NC
    layer_pars.NC *= 2


    fal = FALinear(layer_pars.NCOLD * 2 * 2, pars.num_class, fa=pars.fa)


    aux = nn.Sequential(
        nn.AvgPool2d(int(layer_pars.HW / 2), int(layer_pars.HW / 2)),
        nn.Flatten(),
        fal
    )

    net = nn.Sequential()
    net.add_module('layer', layer)
    net.add_module('aux', aux)

    update_pars(pars,net,fix)


    return net, fix, layer


def get_net_a(pars, fix, layer_pars):
    if (pars.i == 2) or (pars.i == 3):
        fix.add_module('max_pool%d' % pars.i, nn.MaxPool2d(2, stride=2))
        layer_pars.HW /= 2

    layer = nn.Sequential(
        FAConv2d(layer_pars.NCOLD, layer_pars.NC, 3, padding=1, fa=pars.fa, device=pars.device),
        nn.Hardtanh(min_val=0, max_val=pars.MX)
    )
    layer_pars.NCOLD = layer_pars.NC

    fal = FALinear(layer_pars.NC * 4, pars.num_class, fa=pars.fa)
    aux = nn.Sequential(
        nn.AvgPool2d(int(layer_pars.HW / 2), int(layer_pars.HW / 2)),
        nn.Flatten(),
        fal
    )

    net = nn.Sequential()
    net.add_module('layer', layer)
    net.add_module('aux', aux)

    update_pars(pars,net,fix)


    return net, fix, layer


def get_net_pre(pars):

    if pars.CR == 'CE':
        pars.criterion = nn.CrossEntropyLoss()
    else:
        pars.criterion = hinge_loss(num_class=pars.num_class)

