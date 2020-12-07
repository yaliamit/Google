import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from layers import Linear, ident







class encoder_mix(nn.Module):
    def __init__(self,model):
        super(encoder_mix,self).__init__()
        self.n_mix=model.n_mix
        self.x_dim=model.x_dim

        self.only_pi=model.only_pi

        #if not self.only_pi:
        #    self.x2hpi = nn.Linear(model.x_dim, model.h_dim)
        self.h2smu = nn.Linear(model.x_dim, model.s_dim * model.n_mix)
        self.h2svar = nn.Linear(model.x_dim, model.s_dim * model.n_mix, bias=False)
        if not self.only_pi:
            self.h2pi = nn.Linear(model.x_dim, model.n_mix)



    def forward(self,inputs,enc_conv):
        pi=None

        # Run the predesigned network could be just the input
        h=enc_conv.forw(inputs)

        hpi=h
        s_mu = self.h2smu(h.reshape(-1,self.x_dim))
        s_logvar = F.threshold(self.h2svar(h.reshape(-1,self.x_dim)), -6, -6)

        if not self.only_pi:
            hm = self.h2pi(hpi.reshape(-1,self.x_dim)).clamp(-10., 10.)
            pi = torch.softmax(hm, dim=1)

        return s_mu, s_logvar, pi





# Each set of s_dim normals gets multiplied by its own matrix to correlate
class decoder_mix(nn.Module):
    def __init__(self,model,args):
        super(decoder_mix,self).__init__()

        self.z2h=[]
        self.n_mix=model.n_mix
        self.z_dim=model.z_dim
        self.u_dim=model.u_dim
        self.x_dim=model.x_dim
        self.final_shape=model.final_shape
        self.type=model.type
        self.output_cont = model.output_cont

        h_dim_a = self.x_dim
        # Full or diagonal normal dist of next level after sample.
        self.z2z=None; self.u2u=None
        if args.decoder_gaus is not None and 'zz' in args.decoder_gaus:
            self.z2z = nn.ModuleList([Linear(self.z_dim, self.z_dim, args.Diag, args.scale) for i in range(self.n_mix)])
        #self.z2z = nn.ModuleList([nn.Identity() for i in range(self.n_mix)])


        if (self.type == 'tvae' and args.decoder_gaus is not None and 'uu' in args.decoder_gaus):
            self.u2u = nn.ModuleList([nn.Linear(self.u_dim, self.u_dim, bias=False) for i in range(self.n_mix)])
            for ll in self.u2u:
                ll.weight.data.fill_(0.)
        # Same z2h for all mixture components:
        self.z2h=Linear(self.z_dim, h_dim_a)
        #self.z2h = nn.ModuleList([Linear(self.z_dim, h_dim_a,scale=args.scale) for i in range(self.n_mix)])
        self.bnh = nn.Identity() #BatchNorm1d(h_dim_a)

        num_hs=1


        #self.enc_conv=model.enc_conv

        #self.h2x = nn.ModuleList([nn.Linear(h_dim_a, self.x_dim) for i in range(num_hs)])


    def forward(self, s, enc_conv, rng=None):

        if (rng is None):
            rng = range(s.shape[0])
        u = s.narrow(len(s.shape) - 1, 0, self.u_dim)
        z = s.narrow(len(s.shape) - 1, self.u_dim, self.z_dim)
        h=[]; v=[]; hz=[]
        for i,zz,vv in zip(rng,z,u):
            if self.z2z is not None:
                hz+=[self.z2z[i](zz)]
            else:
                hz+=[zz]
            if (self.type=='tvae'):
                if self.u2u is not None:
                    v=v+[self.u2u[i](vv)]
                else:
                    v+=[vv]

        for i, hzz in zip(rng,hz):
            if 'List' in str(type(self.z2h)):
                h += [self.bnh(self.z2h[i](hzz))]
            else:
                h += [self.bnh(self.z2h(hzz))]

        h=torch.stack(h,dim=0)
        h=F.relu(h)
        x = []

        for h_, r in zip(h, rng):
            r_ind = 0
            #xx = self.h2x[r_ind](h_)
            xx=h_.reshape([-1]+list(self.final_shape))
            xx=enc_conv.bkwd(xx)
            xx = xx.reshape(xx.shape[0],-1)
            x += [xx]

        xx = torch.stack(x, dim=0)
        #if not self.output_cont:
        xx = torch.sigmoid(xx)
        # else:
        #     xx = torch.tanh(xx)

        return xx, v













