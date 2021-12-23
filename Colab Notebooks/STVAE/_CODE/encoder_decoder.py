import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from layers import Linear, ident, NONLIN







class encoder_mix(nn.Module):
    def __init__(self,model,args):
        super(encoder_mix,self).__init__()
        #self.n_mix=model.n_mix
        #self.x_dim=model.x_dim

        #self.h2smu = Linear(model.x_dim, model.s_dim * model.n_mix, scale=args.scale)
        #self.h2smu = nn.Identity() # Linear(args.temp.x_dim, args.s_dim * args.n_mix, scale=args.scale)

        #self.h2smu = nn.Linear(model.x_dim, model.s_dim * model.n_mix)
        #self.h2svar = nn.Identity() #nn.Linear(args.temp.x_dim, args.s_dim * args.n_mix, bias=False)
        #self.h2pi = nn.Linear(args.temp.x_dim, args.n_mix, bias=False)
        #self.h2pi.weight.data*=args.h2pi_scale

    def forward(self,inputs,args, enc_conv):
        pi=None

        # Run the predesigned network could be just the input
        h,h1=enc_conv.forw(inputs,args)

        hpi=h
        s_mu = h1['dense_mu'] if 'dense_mu' in h1 else h1['conv_mu']
        s_logvar= h1['dense_var'] if 'dense_var' in h1 else h1 ['conv_var']
        #s_mu = h1['dense_mu'] #self.h2smu(h.reshape(-1,args.temp.x_dim))
        #s_logvar = h1['dense_var'] #F.threshold(self.h2svar(h.reshape(-1,args.temp.x_dim)), -6, -6)

        #hm = self.h2pi(hpi.reshape(-1,args.temp.x_dim)).clamp(-10., 10.)
        pi = torch.softmax(h1['dense_pi'], dim=1)

        return s_mu, s_logvar, pi,[h,h1]





# Each set of s_dim normals gets multiplied by its own matrix to correlate
class decoder_mix(nn.Module):
    def __init__(self,model,args):
        super(decoder_mix,self).__init__()

        self.z2h=[]
        # self.n_mix=model.n_mix
        # self.z_dim=model.z_dim
        # self.u_dim=model.u_dim
        args.temp.u_dim=model.u_dim
        args.temp.z_dim=model.z_dim
        # self.x_dim=model.x_dim
        # self.final_shape=model.final_shape
        # self.type=model.type
        self.decoder_nonlinearity=model.decoder_nonlinearity
        self.penalty=model.penalty
        #h_dim_a = self.x_dim
        # Full or diagonal normal dist of next level after sample.
        self.z2z=None; self.u2u=None
        if args.decoder_gaus is not None and 'zz' in args.decoder_gaus:
            self.z2z = nn.ModuleList([Linear(args.temp.z_dim, args.temp.z_dim, args.Diag, args.scale, args.Iden) for i in range(args.n_mix)])


        #self.z2z = nn.ModuleList([nn.Identity() for i in range(self.n_mix)])

        self.decoder_nonlin=NONLIN(self.decoder_nonlinearity)
        if (self.type == 'tvae' and args.decoder_gaus is not None and 'uu' in args.decoder_gaus):
            self.u2u = nn.ModuleList([nn.Linear(args.temp.u_dim, args.temp.u_dim, bias=False) for i in range(args.n_mix)])
            for ll in self.u2u:
                ll.weight.data.fill_(0.)
            #self.u2ui = nn.ModuleList([nn.Linear(args.temp.u_dim, args.temp.u_dim) for i in range(args.n_mix)])
        # Same z2h for all mixture components:
        #self.z2h=Linear(self.z_dim, h_dim_a)
        self.z2h = nn.ModuleList([Linear(args.temp.z_dim, args.temp.x_dim,scale=args.scale) for i in range(args.n_mix)])
        self.bnh = nn.Identity() #BatchNorm1d(h_dim_a)

        num_hs=1



    def forward(self, s, args, enc_conv, rng=None):

        if (rng is None):
            rng = range(s.shape[0])
        u = s.narrow(len(s.shape) - 1, 0, args.temp.u_dim)
        z = s.narrow(len(s.shape) - 1, args.temp.u_dim, args.temp.z_dim)
        h=[]; v=[]; hz=[]
        if not hasattr(self,'cluster_hidden') or not self.cluster_hidden:
            for i,zz,vv in zip(rng,z,u):
                if self.z2z is not None:
                        hz+=[self.z2z[i](zz)]
                else:
                    hz+=[zz]
                if (args.type=='tvae'):
                    if self.u2u is not None:
                        v=v+[self.u2u[i](vv)]
                    else:
                        v+=[vv]
        else:
            hz=z
            v=u

        for i, hzz in zip(rng,hz):
            if 'List' in str(type(self.z2h)):
                h += [self.bnh(self.z2h[i](hzz))]
            else:
                h += [self.bnh(self.z2h(hzz))]
        h = torch.stack(h, dim=0)
        xx=self.h_to_x(h,args, enc_conv,rng)

        return xx,v

    def h_to_x(self, h,args, enc_conv, rng=None):

        if (rng is None):
            rng = range(h.shape[0])

        h = self.decoder_nonlin(h)
        x = []

        for h_, r in zip(h, rng):
            r_ind = 0
            # xx = self.h2x[r_ind](h_)
            xx = h_.reshape([-1] + list(args.temp.final_shape))
            xx = enc_conv.bkwd(xx)
            xx = xx.reshape(xx.shape[0], -1)
            x += [xx]

        xx = torch.stack(x, dim=0)
        # if not self.output_cont:
        xx = torch.sigmoid(xx)
        # else:
        #     xx = torch.tanh(xx)

        return xx


    def forward_specific(self,S,enc_conv):

        h=self.bnh(self.z2h[0](S[0]))

        xx = h.reshape([-1] + list(self.final_shape))
        xx = enc_conv.bkwd(xx)
        xx = torch.sigmoid(xx.reshape(xx.shape[0], -1))

        return xx







