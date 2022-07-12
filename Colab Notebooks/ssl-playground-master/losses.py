'''
Some of the loss are from https://github.com/C16Mftang/biological-SSL/blob/main/loss.py
'''
import torch
import pytorch_lightning as pl

class SimpleHingeLoss(pl.LightningModule):
    def __init__(self, margin=1.0, normalize=True):
        super().__init__()
        self.margin = margin
        self.normalize = normalize

    def forward(self, x1, x2):
        if self.normalize:
            x1 = torch.nn.functional.normalize(x1, dim=1)
            x2 = torch.nn.functional.normalize(x2, dim=1)
        return torch.sum(torch.relu(torch.abs(x1 - x2) - self.margin))
        

class ContrastiveHinge(pl.LightningModule):
    def __init__(self, thr=2., delta=1., grad_block=True, normalize=True):
        super(ContrastiveHinge, self).__init__()
        self.thr = thr
        self.delta = delta
        self.grad_block = grad_block
        self.normalize = normalize

    def forward(self, out0, out1):
        # print(f'out0: {out0.shape}, out1: {out1.shape}')

        n = out0.shape[0] # half of original n
        if self.grad_block:
            out0 = out0.clone().detach()  # grad block
        if self.normalize:
            out0 = torch.nn.functional.normalize(out0, dim=1)
            out1 = torch.nn.functional.normalize(out1, dim=1)
        out0b = out0.repeat([n, 1])
        out1b = out1.repeat_interleave(n, dim=0)
        # print(f'out0b: {out0b.shape}, out1b: {out1b.shape}')
        outd = out0b - out1b
        outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
        # print(f'outd: {outd.shape}')
        OUT = -outd.reshape(n, n).transpose(0, 1)
        # print(f'OUT: {OUT.shape}')
        # Multiply by y=-1/1
        OUT = (OUT + self.thr) * (2. * torch.eye(n, device=self.device) - 1.)
        # print('mid',time.time()-t1)
        loss = torch.sum(torch.relu(self.delta - OUT))
        # exit(0)
        return loss


class ContrastiveHinge2(pl.LightningModule):
    def __init__(self, thr1=1., thr2=3., grad_block=True, normalize=True):
        super(ContrastiveHinge2, self).__init__()
        self.thr1 = thr1
        self.thr2 = thr2
        self.bmask = None
        self.grad_block = grad_block
        self.normalize = normalize

    def create_bmask(self, batch_size):
        bmask = torch.eye(batch_size, dtype=torch.bool, device=self.device)
        return bmask

    def forward(self, h1, h2):
        n = h1.shape[0]
        if self.bmask is None:
            self.bmask = self.create_bmask(n)
        if self.grad_block:
            h1 = h1.clone().detach()
        if self.normalize:
            h1 = torch.nn.functional.normalize(h1, dim=1)
            h2 = torch.nn.functional.normalize(h2, dim=1)
        h1b = h1.repeat([n, 1])  # bsz*bsz, 64
        h2b = h2.repeat_interleave(n, dim=0)  # bsz*bsz, 64
        outd = h1b - h2b  # bsz*bsz, 64
        # 1-norm
        outd = torch.sum(torch.abs(outd), dim=1)  # bsz*bsz, 1
        # distance matrix, flipped
        OUT = outd.reshape(n, n).transpose(0, 1)  # bsz, bsz
        # Multiply by y=-1/1; 2*eye-1 is a matrix with 1 in the diagonal, -1 off diagonal
        pos = torch.diag(OUT)  # bsz,
        neg = OUT[~self.bmask]  # bsz*(bsz-1),
        loss = torch.sum(torch.relu(pos - self.thr1)) + torch.sum(torch.relu(self.thr2 - neg))
        return loss


class HingeFewerNegs(pl.LightningModule):
    def __init__(self, thr=2., delta=1., future=5, grad_block=True, normalize=True):
        super(HingeFewerNegs, self).__init__()
        self.thr = thr
        self.delta = delta
        self.future = future
        self.mask = None
        self.grad_block = grad_block
        self.normalize = normalize

    def create_bmask(self, batch_size):
        mask = torch.eye(batch_size).to(self.device)
        mask = 2. * mask - 1.
        return mask

    def forward(self, out0, out1):
        n = out0.shape[0]
        if self.mask is None:
            self.mask = self.create_bmask(n)
        if self.grad_block:
            out0 = out0.clone().detach()  # grad block
        if self.normalize:
            out0 = torch.nn.functional.normalize(out0, dim=1)
            out1 = torch.nn.functional.normalize(out1, dim=1)
        out0b = out0.repeat([n, 1])
        out1b = out1.repeat_interleave(n, dim=0)
        outd = out0b - out1b
        outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
        OUT = -outd.reshape(n, n).transpose(0, 1)
        # Multiply by y=-1/1
        OUT = (OUT + self.thr) * self.mask

        if self.future != 0:
            loss = 0
            for i in range(self.future):
                fac = 1. if i == 0 else 1. / self.future
                loss += fac * (torch.sum(torch.relu(self.delta - torch.diagonal(OUT, i))))
        else:
            loss = torch.sum(torch.relu(self.delta - OUT))
        return loss
        

class HingeNoNegs(pl.LightningModule):
    def __init__(self, thr=2., delta=1., grad_block=True, normalize=True):
        super().__init__()
        self.thr = thr
        self.delta = delta
        self.grad_block = grad_block
        self.normalize = normalize

    def forward(self, out0, out1):
        if self.grad_block:
            out0 = out0.clone().detach()  # grad block
        if self.normalize:
            out0 = torch.nn.functional.normalize(out0, dim=1)
            out1 = torch.nn.functional.normalize(out1, dim=1)
        diff = torch.sum(torch.abs(out0 - out1), dim=1)
        diff = -diff + self.thr
        return torch.sum(torch.relu(self.delta - diff))