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
        out_a = outa / (sd + .0001)
        #print(sd.mean(),sd.std())
    return out_a

class AE_loss(nn.Module):
    def __init__(self, lamda=0, l1=False, double_aug=False):
        super(AE_loss, self).__init__()

        self.lamda=lamda
        if l1:
            self.criterion = nn.L1Loss()
        else:
            self.criterion=nn.MSELoss()

        self.double_aug=double_aug

    def forward(self,out0,out1,emb0,emb1,data0,data1):

        dim_emb=emb0.shape[1]
        loss=self.criterion(out1,data0)

        emb_loss=torch.tensor([0.])
        if self.lamda>0:
            emb_loss=self.lamda*self.criterion(emb0,emb1)
            loss+=emb_loss

        return loss, emb_loss



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

class direct_loss(nn.Module):
    def __init__(self, batch_size, out_dim, eps=0.1, alpha=0.9, lamda=0.,device='cpu'):
        super(direct_loss, self).__init__()
        self.dv = device
        self.eps=eps
        self.alpha=alpha
        self.cov=torch.eye(out_dim).to(self.dv)
        self.eye=self.eps*torch.eye(out_dim).to(self.dv)
        self.lamda=lamda
        self.batch_size=batch_size

    def forward(self,out0,out1):


        with torch.no_grad():
            self.cov=(1-self.alpha)*(out1.T @ out1)/self.batch_size+self.alpha*self.cov

        outa=out1 @ (self.cov + self.eye)
        diff=torch.sum(torch.abs(out0-outa),dim=1)
        #print(torch.max(diff),torch.min(diff))
        loss= torch.sum(torch.relu(diff)) #torch.sum(torch.abs(outa-out0)) #+self.lamda*(torch.mean(.5-torch.abs(out0)))
        loss1=torch.tensor([0.])
        # if self.lamda>0:
        #      loss1= torch.sum(torch.relu(torch.sum(torch.abs(out0-out1),dim=1)-1.))
        #      loss+=self.lamda*loss1

        return loss, loss1


class barlow_loss(nn.Module):
    def __init__(self, batch_size, dim, device='cpu', lamda=.004, scale=1./32.):
        super(barlow_loss, self).__init__()
        self.batch_size = batch_size # 2000 in my experiments
        self.device = device
        self.lamda = lamda # 3.9e-3 in the paper
        self.scale = scale # 1/32 in the paper

    def off_diagonal(self,c):
        return c-torch.diag_embed(torch.diagonal(c))

    def forward(self, out0, out1):
        # two branches
        out0a = out0-torch.mean(out0,dim=0,keepdim=True)
        out1a = out1-torch.mean(out1,dim=0,keepdim=True)
        out0a = out0a/(torch.sqrt(torch.mean(out0a*out0a,dim=0,keepdim=True))+.0001)
        out1a = out1a/(torch.sqrt(torch.mean(out1a*out1a,dim=0,keepdim=True))+.0001)

        # empirical cross-correlation matrix
        c = (out0a).T @ (out1a)
        c.div_(self.batch_size)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.scale)
        off_diag = self.off_diagonal(c).pow_(2).sum().mul(self.scale)
        loss = on_diag + self.lamda * off_diag
        return loss, off_diag

class simclr_loss(nn.Module):
    def  __init__(self,dv, bsz, tau=1.):
        super(simclr_loss,self).__init__()
        self.dv=dv
        self.tau=tau
        self.bsz=bsz
        self.ID=2*torch.eye(bsz).to(dv)-1

    def __call__(self,out0,out1):

        # Standardize 64 dim outputs of original and deformed images
        out0a = standardize(out0,False)
        out1a = standardize(out1,False)
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
        lecov = .5 * (lecov) - v

        loss = torch.mean(lecov)
        # Accuracy

        icov = self.ID * COV
        acc = torch.sum((icov > -.5).type(torch.float)) / self.bsz

        return loss, acc



class binary_loss(nn.Module):
    def  __init__(self,dv, nostd=False, thr=.75):
        super(binary_loss,self).__init__()
        self.dv=dv
        self.nostd=nostd
        self.thr=thr

    def __call__(self,out0, out1):

        bsz=out0.shape[0]
        # Standardize 64 dim outputs of original and deformed images
        out0a = standardize(out0, self.nostd)
        out1a = standardize(out1, self.nostd)
        # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
        COV = torch.mm(out0a, out1a.transpose(0, 1))
        cc=COV.flatten()
        targ=torch.zeros(cc.shape[0]).to(self.dv)
        targ[0:cc.shape[0]:(COV.shape[0]+1)]=1
        loss=F.binary_cross_entropy_with_logits(cc,targ,pos_weight=torch.tensor(float(bsz)))


        icov = (cc-self.thr) * (2.*targ-1.)
        acc = torch.sum((icov > 0).type(torch.float)) / bsz

        return loss, acc



class L1_loss(nn.Module):
    def  __init__(self,dv,bsz, future=0, thr=2., delta=1., WW=1., nostd=True):
        super(L1_loss,self).__init__()
        self.dv=dv
        self.bsz=bsz
        self.future=future
        self.thr=thr
        self.delta=delta
        self.WW=WW
        self.nostd=nostd

    def __call__(self,out0,out1):
        out0 = standardize(out0, self.nostd)
        out1 = standardize(out1, self.nostd)
        bsz=out0.shape[0]
        CC=-torch.cdist(out0,out1,p=1)
        OUT = -(self.thr+CC)
        diag=-torch.diag(OUT)
        OUT[range(bsz),range(bsz)]=diag

        if self.future:
            loss = 0
            for i in range(self.future):
                # fac = 1. if i==0 else 1./future
                loss += (torch.sum(torch.relu(self.delta - torch.diagonal(OUT, i))))
        elif self.future == 0:
            loss = (1 - self.WW) * torch.sum(torch.relu(self.delta - torch.diag(OUT))) + self.WW * torch.sum(torch.relu(self.delta - OUT))
            # loss = torch.sum(torch.relu(delta - OUT))

        acc = torch.sum(OUT > 0).type(torch.float) / bsz

        return loss, acc


class clapp_loss(nn.Module):
    def  __init__(self,dv, delta=1., future=0, nostd=True):
        super(clapp_loss,self).__init__()

        self.dv=dv
        self.delta=delta
        self.future=future
        self.nostd=nostd

    def __call__(self,out0, out1):

        bsz = out0.shape[0]
        out0=out0.reshape(bsz, -1)
        out1=out1.reshape(bsz, -1)
        out0 = standardize(out0,self.nostd)
        out1 = standardize(out1,self.nostd)
        OUT=torch.mm(out0,out1.transpose(0,1))

        # Multiply by y=-1/1
        OUT = OUT  * (2. * torch.eye(bsz).to(self.dv) - 1.)

        if self.future:
            loss=0
            for i in range(self.future):
                fac = 1. if i==0 else 1./self.future
                loss+=fac*(torch.sum(torch.relu(self.delta-torch.diagonal(OUT,i))))
        elif self.future==0:
            loss = torch.sum(torch.relu(self.delta - OUT))

        acc = torch.sum(OUT > 0).type(torch.float) / bsz


        return loss, acc