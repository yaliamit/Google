import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np



class biass(nn.Module):
    def __init__(self,dim, scale=None):
        super(biass,self).__init__()

        self.dim=dim
        if (scale is None):
            self.bias=nn.Parameter(6*(torch.rand(self.dim) - .5)/ np.sqrt(self.dim))
        else:
            self.bias=nn.Parameter(scale*(torch.rand(self.dim)-.5))



    def forward(self,z):
        return(self.bias.repeat(z.shape[0],1))

class ident(nn.Module):
    def __init__(self):
        super(ident,self).__init__()

    def forward(self,z):
        return(torch.ones(z.shape[0]))


class diag(nn.Module):
    def __init__(self,dim):
        super(diag,self).__init__()
        self.dim = dim
        if (dim>0):
            rim=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
            self.mu=nn.Parameter(rim)
            ris=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
            self.sigma=nn.Parameter(ris)

    def forward(self,z):
        u=z*self.sigma+self.mu

        return(u)

class Linear(nn.Module):
    def __init__(self, dim1,dim2,diag_flag=False, scale=None):
        super(Linear, self).__init__()

        # If dimensions are zero just return a dummy variable of the same dimension as input
        self.lin=ident()
        # If diagonal normal with diagonal cov.
        if (diag_flag and dim1>0):
            self.lin=diag(dim1)
        else:
            if (dim2>0):
                if (dim1>0):
                    bis = True if dim1>1 else False
                    self.lin=nn.Linear(dim1,dim2,bias=bis)
                    if scale is not None:
                        self.lin.weight.data*=scale
                # Only a bias term that does not depend on input.
                else:
                    self.lin=biass(dim2, scale)

    def forward(self,z):
        return self.lin(z)


class iden_copy(nn.Module):
    def __init__(self):
        super(iden_copy,self).__init__()

    def forward(self,z):
        return(z)



class encoder_mix(nn.Module):
    def __init__(self,model):
        super(encoder_mix,self).__init__()
        self.n_mix=model.n_mix
        self.x_dim=model.x_dim

        self.feats=model.feats
        self.feats_back=model.feats_back
        self.only_pi=model.only_pi

        if not self.only_pi:
            self.x2hpi = nn.Linear(model.x_dim, model.h_dim)
        self.h2smu = nn.Linear(model.h_dim, model.s_dim * model.n_mix)
        self.h2svar = nn.Linear(model.h_dim, model.s_dim * model.n_mix, bias=False)
        if not self.only_pi:
            self.h2pi = nn.Linear(model.h_dim, model.n_mix)

        #self.enc_conv=model.enc_conv

        self.x2h = nn.Linear(model.x_dim, model.h_dim)

    def forward(self,inputs,enc_conv):
        pi=None

        # Run the predesigned network could be just the input
        out=enc_conv.forw(inputs)
        inputs=out.reshape(-1, self.x_dim)

        # One more fully connected layer
        h = F.relu(self.x2h(inputs))
        if not self.only_pi:
            hpi = F.relu(self.x2hpi(inputs))
        # Then connect to means and variances
        s_mu = self.h2smu(h)
        s_logvar = F.threshold(self.h2svar(h), -6, -6)

        if not self.only_pi:
            hm = self.h2pi(hpi).clamp(-10., 10.)
            pi = torch.softmax(hm, dim=1)

        return s_mu, s_logvar, pi





# Each set of s_dim normals gets multiplied by its own matrix to correlate
class decoder_mix(nn.Module):
    def __init__(self,model,args):
        super(decoder_mix,self).__init__()

        self.z2h=[]
        self.n_mix=model.n_mix
        self.z_dim=model.z_dim
        self.h_dim=model.h_dim
        self.u_dim=model.u_dim
        self.x_dim=model.x_dim
        self.final_shape=model.final_shape
        self.feats=model.feats
        self.feats_back=model.feats_back
        self.type=model.type
        self.output_cont = model.output_cont

        h_dim_a = self.h_dim

        # Full or diagonal normal dist of next level after sample.

        #self.z2z = nn.ModuleList([Linear(self.z_dim, self.z_dim, args.Diag) for i in range(self.n_mix)])
        self.z2z = nn.ModuleList([nn.Identity() for i in range(self.n_mix)])


        if (self.type == 'tvae'):
            self.u2u = nn.ModuleList([nn.Linear(self.u_dim, self.u_dim, bias=False) for i in range(self.n_mix)])
            for ll in self.u2u:
                ll.weight.data.fill_(0.)

        self.z2h = nn.ModuleList([Linear(self.z_dim, h_dim_a,scale=args.scale) for i in range(self.n_mix)])
        self.bnh = nn.Identity() #BatchNorm1d(h_dim_a)

        num_hs=1


        #self.enc_conv=model.enc_conv

        self.h2x = nn.ModuleList([nn.Linear(h_dim_a, self.x_dim) for i in range(num_hs)])


    def forward(self, s, enc_conv, rng=None):

        if (rng is None):
            rng = range(s.shape[0])
        u = s.narrow(len(s.shape) - 1, 0, self.u_dim)
        z = s.narrow(len(s.shape) - 1, self.u_dim, self.z_dim)
        h=[]; v=[]
        for i,zz,vv in zip(rng,z,u):
            zzt=self.z2z[i](zz)
            h+=[self.bnh(self.z2h[i](zzt))]
            if (self.type=='tvae'):
                v=v+[self.u2u[i](vv)]

        h=torch.stack(h,dim=0)
        h=F.relu(h)
        x = []

        for h_, r in zip(h, rng):
            r_ind = 0
            xx = self.h2x[r_ind](h_)
            xx=xx.reshape([-1]+list(self.final_shape))
            xx=enc_conv.bkwd(xx)
            xx = xx.reshape(xx.shape[0],-1)
            x += [xx]

        xx = torch.stack(x, dim=0)
        #if not self.output_cont:
        xx = torch.sigmoid(xx)
        # else:
        #     xx = torch.tanh(xx)

        return xx, v













