import torch
import torch.nn as nn

def standardize(out):
    outa = out.reshape(out.shape[0], -1)  # -torch.mean(out,dim=1).reshape(-1,1)
    # out_a = torch.sign(outa) / out.shape[1]
    sd = torch.sqrt(torch.sum(outa * outa, dim=1)).reshape(-1, 1)
    out_a = outa / (sd + .01)

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

def get_embedd_loss(out0,out1,dv):

        bsz=out0.shape[0]
        out0a=standardize(out0)
        out1a=standardize(out1)
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

        acc=torch.sum((icov>0).type(torch.float))/bsz
        return loss,acc


def get_embedd_loss_binary(out0, out1,dv):

        bsz=out0.shape[0]
        # Standardize 64 dim outputs of original and deformed images
        out0a = standardize(out0)
        out1a = standardize(out1)
        # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
        COV = torch.mm(out0a, out1a.transpose(0, 1))
        cc=COV.flatten()
        targ=torch.zeros(cc.shape[0]).to(dv)
        targ[0:cc.shape[0]:(COV.shape[0]+1)]=1
        loss=F.binary_cross_entropy_with_logits(cc,targ,pos_weight=torch.tensor(float(bsz)))


        icov = (cc-.75) * (2.*targ-1.)
        acc = torch.sum((icov > 0).type(torch.float)) / bsz

        return loss, acc


def get_embedd_loss_new_a(out0, out1,dv):

        thr1=.9
        thr2=.3
        bsz=out0.shape[0]
        thr=(thr1+thr2)*.5
        # Standardize 64 dim outputs of original and deformed images
        out0a = standardize(out0)
        out1a = standardize(out1)
        # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
        #COV = torch.mm(out0a, out1a.transpose(0, 1))
        COV = torch.mm(out0a, out1a.transpose(0, 1))
        cc = COV.flatten()
        targ = torch.zeros(cc.shape[0],dtype=torch.bool).to(dv)
        targ[0:cc.shape[0]:(COV.shape[0] + 1)] = 1
        cc1=torch.relu(thr1-cc[targ])
        cc2=torch.relu(cc[targ.logical_not()]-thr2)
        loss1=torch.sum(cc1)
        loss2=torch.sum(cc2)
        loss=(loss1+loss2)/(bsz*bsz)

        acc = (torch.sum(cc[targ]>thr)+torch.sum(cc[targ.logical_not()]<thr)).type(torch.float) / bsz

        return loss, acc



def get_embedd_loss_new(out0, out1, dv):
    thr = 2.
    bsz = out0.shape[0]
    # out0=torch.tanh(out0)
    out0 = standardize(out0)
    # out1=torch.tanh(out1)
    out1 = standardize(out1)
    out0b = out0.repeat([bsz, 1])
    out1b = out1.repeat_interleave(bsz, dim=0)
    outd = out0b - out1b
    outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
    OUT = -outd.reshape(bsz, bsz).transpose(0, 1)
    # Multiply by y=-1/1
    OUT = (OUT + thr) * (2. * torch.eye(bsz).to(dv) - 1.)

    loss = torch.sum(torch.relu(1 - OUT))

    acc = torch.sum(OUT > 0).type(torch.float) / bsz

    return loss, acc