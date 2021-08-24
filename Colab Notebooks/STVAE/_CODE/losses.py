import torch
import torch.nn as nn
import torch.nn.functional as F
import time
def standardize(out, nostd):

    if nostd:
        return out
    else:
        outa = out.reshape(out.shape[0], -1)
        sd = torch.sqrt(torch.sum(outa * outa, dim=1)).reshape(-1, 1)
        out_a = outa / (sd + .01)
        #print(sd.mean(),sd.std())
    return out_a


class hinge_loss(nn.Module):
    def __init__(self, mu=1., num_class=10):
        super(hinge_loss, self).__init__()
        self.fac = mu/(num_class-1)
        self.ee = torch.eye(num_class)

    def forward(self, input, target):

        targarr = self.ee[target] > 0
        loss = torch.sum(torch.relu(1 - input[targarr])) + self.fac * torch.sum(
            torch.relu(1 + input[torch.logical_not(targarr)]))
        loss /= input.shape[0]
        return loss


class Barlow_loss(nn.Module):
    def __init__(self, batch_size, device, lambd=.004, scale=1./32.):
        super(Barlow_loss, self).__init__()
        self.batch_size = batch_size # 2000 in my experiments
        self.device = device
        self.lambd = lambd # 3.9e-3 in the paper
        self.scale = scale # 1/32 in the paper

        # normalization layer for the representations x1 and x2
        # '64' here refers to the output dimension of the base encoder
        # the paper claimed that this loss benefits from larger output dim
        # I have tried dim=512 and 1024, no significant differences
        self.bn = nn.BatchNorm1d(64, affine=False, track_running_stats=True)
    def off_diagonal(self,c):
        return c-torch.diag_embed(torch.diagonal(c))

    def forward(self, X):
        # two branches
        x1 = X[::2]
        x2 = X[1::2]

        # empirical cross-correlation matrix
        c = self.bn(x1).T @ self.bn(x2)
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale)
        loss = on_diag + self.lambd * off_diag
        return loss

def simclr_loss(out0, out1, dv, nostd):
        # Standardize 64 dim outputs of original and deformed images
        bsz=out0.shape[0]
        out0a = standardize(out0,nostd)
        out1a = standardize(out1,nostd)
        # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
        COV = torch.mm(out0a, out1a.transpose(0, 1))
        COV1 = torch.mm(out1a, out1a.transpose(0, 1))
        COV0 = torch.mm(out0a, out0a.transpose(0, 1))
        # Diagonals of covariances.
        v0 = torch.diag(COV0)
        v1 = torch.diag(COV1)
        v = torch.diag(COV)
        # Mulitnomial logistic loss just computed on positive match examples, with all other examples as a separate class.
        lecov = torch.log(
            torch.exp(torch.logsumexp(COV, dim=1)) + torch.exp(torch.logsumexp(COV0 - torch.diag(v0), dim=1)))
        lecov += torch.log(
            torch.exp(torch.logsumexp(COV, dim=1)) + torch.exp(torch.logsumexp(COV1 - torch.diag(v1), dim=1)))
        lecov = (.5 * (lecov) - v)

        loss = torch.mean(lecov)
        # Accuracy
        ID = 2. * torch.eye(out0.shape[0]).to(dv) - 1.
        icov = ID * COV
        acc = torch.sum((icov > -.5).type(torch.float)) / bsz

        return loss, acc


def get_embedd_loss(out0,out1,dv, nostd):

        bsz=out0.shape[0]
        out0a=standardize(out0, nostd)
        out1a=standardize(out1, nostd)
        COV=torch.mm(out0a,out1a.transpose(0,1))
        COV1 = torch.mm(out1a, out1a.transpose(0, 1))
        COV0 = torch.mm(out0a,out0a.transpose(0,1))
        vb=(torch.eye(bsz)*1e10).to(dv)

        cc = torch.cat((COV, COV0 - vb), dim=1)
        targ = torch.arange(bsz).to(dv)
        l1 = F.cross_entropy(cc, targ)
        cc = torch.cat((COV.T, COV1 - vb), dim=1)
        l2 = F.cross_entropy(cc, targ)
        loss =  (l1 + l2) / 2

        ID=2.*torch.eye(out0.shape[0]).to(dv)-1.
        icov=ID*COV

        acc=torch.sum((icov> -.5).type(torch.float))/bsz
        return loss,acc


def get_embedd_loss_binary(out0, out1,dv, nostd):

        bsz=out0.shape[0]
        # Standardize 64 dim outputs of original and deformed images
        out0a = standardize(out0, nostd)
        out1a = standardize(out1, nostd)
        # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
        COV = torch.mm(out0a, out1a.transpose(0, 1))
        cc=COV.flatten()
        targ=torch.zeros(cc.shape[0]).to(dv)
        targ[0:cc.shape[0]:(COV.shape[0]+1)]=1
        loss=F.binary_cross_entropy_with_logits(cc,targ,pos_weight=torch.tensor(float(bsz)))


        icov = (cc-.75) * (2.*targ-1.)
        acc = torch.sum((icov > 0).type(torch.float)) / bsz

        return loss, acc


def get_embedd_loss_future(out0, out1,nostd,future):
    thr = 2.
    #out0 = standardize(out0,nostd)
    #out1 = standardize(out1,nostd)
    loss=torch.sum(torch.relu(1-thr+torch.sum(torch.abs(out0-out1),dim=1)))
    for i in range(1,future):
        loss+=torch.sum(torch.relu(1+thr-torch.sum(torch.abs(out0[0:-i]-out1[i:]),dim=1)))

    return loss

def get_embedd_loss_new(out0, out1, dv, nostd,future=0, thr=4.,delta=2.):
    bsz = out0.shape[0]
    # out0=torch.tanh(out0)
    out0 = standardize(out0,nostd)
    # out1=torch.tanh(out1)
    out1 = standardize(out1,nostd)
    out0b = out0.repeat([bsz, 1])
    out1b = out1.repeat_interleave(bsz, dim=0)
    outd = out0b - out1b
    outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
    OUT = -outd.reshape(bsz, bsz).transpose(0, 1)


    # Multiply by y=-1/1
    OUT = (OUT + thr) * (2. * torch.eye(bsz).to(dv) - 1.)
    #print('mid',time.time()-t1)

    if future:
        loss=0
        for i in range(future):
            fac = 1. if i==0 else 1./future
            loss+=fac*(torch.sum(torch.relu(delta-torch.diagonal(OUT,i))))
    elif future==0:
        loss = torch.sum(torch.relu(delta - OUT))
    else:
        loss = torch.mean(torch.relu(delta-OUT))+\
               (1-1/(bsz*bsz))*torch.sum(torch.relu(delta-torch.diagonal(OUT,0)))

    

    acc = torch.sum(OUT > 0).type(torch.float) / bsz


    return loss, acc

