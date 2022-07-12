import pytorch_lightning as pl
import torch
from lightly.loss import NTXentLoss
from lightly.loss import NegativeCosineSimilarity
from lightly.models._momentum import _MomentumEncoderMixin
from lightly.models.modules import BYOLProjectionHead
from lightly.models.modules import SimSiamPredictionHead
from lightly.models.modules import SimSiamProjectionHead
from lightly.models.modules.heads import ProjectionHead
from lightly.models.utils import deactivate_requires_grad
from torch import nn
import torch.nn.functional as F

from losses import HingeNoNegs


def inference_backbone_output_shape(backbone):
    device = next(backbone.parameters()).device
    val = torch.randn(2, 3, 32, 32).to(device)
    _, backbone_output_shape = backbone(val).flatten(start_dim=1).shape
    return backbone_output_shape


class BYOLBPProjectionHead(ProjectionHead):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int):
        super().__init__([
            (input_dim, hidden_dim, None, nn.Hardtanh()),
            (hidden_dim, output_dim, None, None),
        ])


class SimSiam(pl.LightningModule):
    def __init__(self, backbone, max_epochs, loss=None):
        super().__init__()
        self.backbone = backbone
        backbone_out_dims = inference_backbone_output_shape(backbone)
        self.projection_head = SimSiamProjectionHead(backbone_out_dims, 64, 64)
        self.prediction_head = SimSiamPredictionHead(64, 32, 64)
        if not loss:
            self.criterion = NegativeCosineSimilarity()
        else:
            self.criterion = loss
        self.max_epochs = max_epochs

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [scheduler]


class SimCLR(pl.LightningModule):
    def __init__(self, backbone, max_epochs, loss=None):
        super().__init__()
        self.backbone = backbone
        backbone_out_dims = inference_backbone_output_shape(backbone)
        # self.projection_head = SimCLRProjectionHead(backbone_out_dims, 2048, 2048)
        self.projection_head = nn.Linear(backbone_out_dims, 64)
        if not loss:
            self.criterion = NTXentLoss()
        else:
            self.criterion = loss
        self.max_epochs = max_epochs

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [scheduler]


class BYOL(pl.LightningModule, _MomentumEncoderMixin):
    def __init__(self, backbone, max_epochs, loss=None):
        super().__init__()
        self.backbone = backbone
        backbone_out_dims = inference_backbone_output_shape(backbone)
        self.projection_head = BYOLProjectionHead(backbone_out_dims, 2048, 2048)
        self.prediction_head = BYOLProjectionHead(2048, 512, 2048)
        self.momentum_backbone = None
        self.momentum_projection_head = None
        self._init_momentum_encoder()
        self.m = 0.9
        if not loss:
            self.criterion = NegativeCosineSimilarity()
        else:
            self.criterion = loss
        self.max_epochs = max_epochs

    def forward(self, x0, x1):
        self._momentum_update(self.m)

        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)
        p0 = self.prediction_head(z0)

        with torch.no_grad():
            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            z1 = self.momentum_projection_head(f1)

        return p0, z1

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        p0, z1 = self.forward(x0, x1)
        p1, z0 = self.forward(x1, x0)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [scheduler]


class DirectCopy(pl.LightningModule, _MomentumEncoderMixin):
    def __init__(self, backbone, max_epochs, loss=None, cm_grad=True,
                 m=0.996, mu=0.5, epsilon=0.3):
        super().__init__()
        self.backbone = backbone
        backbone_out_dims = inference_backbone_output_shape(backbone)
        self.projection_head = BYOLProjectionHead(backbone_out_dims, 256, 64)
        self.momentum_backbone = None
        self.momentum_projection_head = None
        self._init_momentum_encoder()
        self.m = m
        self.mu = mu
        self.epsilon = epsilon
        self.cm_grad = cm_grad

        if not loss:
            self.criterion = NegativeCosineSimilarity()
        else:
            self.criterion = loss
        self.max_epochs = max_epochs
        self.register_buffer('corr_matrix', torch.zeros(64, 64))
        self.register_buffer('eye', torch.eye(64))

    def forward(self, x0, x1):
        self._momentum_update(self.m)

        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)

        if self.cm_grad:
            corr_matrix_batch = torch.einsum('bi,bj->bij', z0, z0).mean(axis=0)
        else:
            with torch.no_grad():
                corr_matrix_batch = torch.einsum('bi,bj->bij', z0, z0).mean(axis=0)
        with torch.no_grad():
            self.corr_matrix = self.mu * self.corr_matrix + (1 - self.mu) * corr_matrix_batch
        # no normalization applied as no performance difference was noted in the paper
        prediction_weights = self.corr_matrix + self.epsilon * self.eye
        p0 = torch.mm(z0, prediction_weights)

        with torch.no_grad():
            f1 = self.momentum_backbone(x1).flatten(start_dim=1)
            z1 = self.momentum_projection_head(f1)

        return p0, z1

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        p0, z1 = self.forward(x0, x1)
        p1, z0 = self.forward(x1, x0)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, self.max_epochs
        )
        return [optim], [scheduler]

def deform_data(x_in, perturb, dv, trans=None, s_factor=4., h_factor=.2):

        if perturb == 0:
            return x_in
        # t1=time.time()
        h = x_in.shape[2]
        w = x_in.shape[3]
        nn = x_in.shape[0]
        v = ((torch.rand(nn, 6) - .5) * perturb).to(dv)
        # v=(torch.rand(nn, 6) * perturb)+perturb/4.
        # vs=2*(torch.rand(nn,6)>.5)-1
        # v=v*vs
        rr = torch.zeros(nn, 6).to(dv)
        u = v
        # Ammplify the shift part of the
        u[:, [2, 5]] *= 2.

        rr[:, [0, 4]] = 1
        if trans is not None:
            if trans == 'shift':
                u[:, [0, 1, 3, 4]] = 0
                u[:, [2, 5]] = torch.tensor([perturb, 0])
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
        x_out = F.grid_sample(x_in, grid, padding_mode='border', align_corners=True)

        if x_in.shape[1] == 3 and s_factor > 0:
            v = torch.rand(nn, 2).to(dv)
            vv = torch.pow(2, (v[:, 0] * s_factor - s_factor / 2)).reshape(nn, 1, 1)
            uu = ((v[:, 1] - .5) * h_factor).reshape(nn, 1, 1)
            x_out_hsv = rgb_to_hsv(x_out, dv)
            x_out_hsv[:, 1, :, :] = torch.clamp(x_out_hsv[:, 1, :, :] * vv, 0., 1.)
            x_out_hsv[:, 0, :, :] = torch.remainder(x_out_hsv[:, 0, :, :] + uu, 1.)
            x_out = hsv_to_rgb(x_out_hsv, dv)
        if trans != 'shift':
            ii = torch.where(torch.bernoulli(torch.ones(nn) * .5) == 1)
            for i in ii:
                x_out[i] = x_out[i].flip(3)

        # print('Def time',time.time()-t1)
        return x_out


def rgb_to_hsv(input, dv):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    mx, inmx = torch.max(input, dim=1)
    mn, inmc = torch.min(input, dim=1)
    df = mx - mn
    h = torch.zeros(input.shape[0], 1).to(dv)
    # if False: #'xla' not in device.type:
    #     h.to(device)
    ii = [0, 1, 2]
    iid = [[1, 2], [2, 0], [0, 1]]
    shift = [360, 120, 240]

    for i, id, s in zip(ii, iid, shift):
        logi = (df != 0) & (inmx == i)
        h[logi, 0] = \
            torch.remainder((60 * (input[logi, id[0]] - input[logi, id[1]]) / df[logi] + s), 360)

    s = torch.zeros(input.shape[0], 1).to(dv)  #
    # if False: #'xla' not in device.type:
    #     s.to(device)
    s[mx != 0, 0] = (df[mx != 0] / mx[mx != 0]) * 100

    v = mx.reshape(input.shape[0], 1) * 100

    output = torch.cat((h / 360., s / 100., v / 100.), dim=1)

    output = output.reshape(sh).transpose(1, 3)
    return output


def hsv_to_rgb(input, dv):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    hh = input[:, 0]
    hh = hh * 6
    ihh = torch.floor(hh).type(torch.int32)
    ff = (hh - ihh)[:, None];
    v = input[:, 2][:, None]
    s = input[:, 1][:, None]
    p = v * (1.0 - s)
    q = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)));

    output = torch.zeros_like(input).to(dv)  # .to(device)
    # if False: #'xla' not in device.type:
    #     output.to(device)
    output[ihh == 0, :] = torch.cat((v[ihh == 0], t[ihh == 0], p[ihh == 0]), dim=1)
    output[ihh == 1, :] = torch.cat((q[ihh == 1], v[ihh == 1], p[ihh == 1]), dim=1)
    output[ihh == 2, :] = torch.cat((p[ihh == 2], v[ihh == 2], t[ihh == 2]), dim=1)
    output[ihh == 3, :] = torch.cat((p[ihh == 3], q[ihh == 3], v[ihh == 3]), dim=1)
    output[ihh == 4, :] = torch.cat((t[ihh == 4], p[ihh == 4], v[ihh == 4]), dim=1)
    output[ihh == 5, :] = torch.cat((v[ihh == 5], p[ihh == 5], q[ihh == 5]), dim=1)

    output = output.reshape(sh)
    output = output.transpose(1, 3)
    return output


class DirectCopyBP(pl.LightningModule):
    def __init__(self, backbone, max_epochs, device, loss=None, cm_grad=False,
                 m=0.996, mu=0.5, epsilon=0.3, perturb=None, symmetric=True, double_aug=True):
        super().__init__()
        self.backbone = backbone
        backbone_out_dims = inference_backbone_output_shape(backbone)
        self.projection_head = BYOLBPProjectionHead(backbone_out_dims, 256, 64)
        self.m = m
        self.mu = mu
        self.epsilon = epsilon
        self.cm_grad = cm_grad
        self.perturb = perturb
        print('in Drc',device)
        self.dv = device
        self.symmetric=symmetric
        self.double_aug=double_aug
        if not loss:
            self.criterion = HingeNoNegs(normalize=False)
        else:
            self.criterion = loss
        self.max_epochs = max_epochs
        self.register_buffer('corr_matrix', torch.zeros(64, 64))
        self.register_buffer('eye', torch.eye(64))

        self.history_size = 2048
        self.z0_history = torch.zeros(self.history_size, 64)
        self.cur_idx = 0

    def forward(self, x0, x1):
        f0 = self.backbone(x0).flatten(start_dim=1)
        z0 = self.projection_head(f0)

        if self.cm_grad:
            corr_matrix_batch = torch.einsum('bi,bj->bij', z0, z0).mean(axis=0)
        else:
            with torch.no_grad():
                corr_matrix_batch = (z0.T @ z0)/z0.shape[0]
                #corr_matrix_batch = torch.einsum('bi,bj->bij', z0, z0).mean(axis=0)
        with torch.no_grad():
            for i in z0:
                self.z0_history[self.cur_idx] = i
                self.cur_idx += 1
                if self.cur_idx == self.history_size:
                    self.cur_idx = 0
            batch_std = torch.std(self.z0_history, axis=0)
            self.log("soft_collapse_ratio", 1 - torch.sum(torch.min(batch_std, torch.tensor(0.1))) / (64 * 0.1))
            self.log("svd_collapse_ratio",
                     1 - torch.sum(torch.min(torch.svd(self.z0_history)[1], torch.tensor(1.))) / 64)
            self.log("cov_collapse_ratio", torch.sum(torch.log(torch.svd(torch.cov(self.z0_history.T))[1]) < -10) / 64)

        with torch.no_grad():
            self.corr_matrix = self.mu * self.corr_matrix + (1 - self.mu) * corr_matrix_batch
        # no normalization applied as no performance difference was noted in the paper
        prediction_weights = self.corr_matrix + self.epsilon * self.eye
        p0 = torch.mm(z0, prediction_weights)

        with torch.no_grad():
            f1 = self.backbone(x1).flatten(start_dim=1)
            z1 = self.projection_head(f1)

        return p0, z1

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        if self.perturb is not None:
            with torch.no_grad():
                x0=deform_data(x0,self.perturb, self.dv)
                if self.double_aug:
                    x1 = deform_data(x1, self.perturb, self.dv)

        p0, z1 = self.forward(x0, x1)
        if self.symmetric:
            p1, z0 = self.forward(x1, x0)
        #
        loss = self.criterion(z1, p0)
        if self.symmetric:
            loss += self.criterion(z1, p0)
            loss*=.5
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [optim]


class LinearProbingClassifier(pl.LightningModule):
    def __init__(self, backbone, max_epochs=100):
        super().__init__()
        # use the pretrained ResNet backbone
        self.backbone = backbone
        self.max_epochs = max_epochs

        # freeze the backbone
        deactivate_requires_grad(backbone)

        backbone_out_dims = inference_backbone_output_shape(backbone)

        # create a linear layer for our downstream classification model
        self.fc = nn.Linear(backbone_out_dims, 10)

        self.criterion = nn.CrossEntropyLoss()
        self.best_acc = 0
        self.best_epoch = 0
        self.last_acc = 0

    def forward(self, x):
        y_hat = self.backbone(x).flatten(start_dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss_fc", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self.forward(x)
        y_hat = torch.nn.functional.softmax(y_hat, dim=1)

        # calculate number of correct predictions
        _, predicted = torch.max(y_hat, 1)
        num = predicted.shape[0]
        correct = (predicted == y).float().sum()
        return num, correct

    def validation_epoch_end(self, outputs):
        # calculate and log top1 accuracy
        if outputs:
            total_num = 0
            total_correct = 0
            for num, correct in outputs:
                total_num += num
                total_correct += correct
            acc = total_correct / total_num
            self.log("val_acc", acc, on_epoch=True, prog_bar=True)
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_epoch = self.current_epoch
            self.last_acc = acc

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [optim]
