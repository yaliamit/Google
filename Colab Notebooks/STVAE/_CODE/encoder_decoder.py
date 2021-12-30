import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from layers import Linear, ident, NONLIN
import network
from get_net_text import get_network




class encoder_mix(nn.Module):
    def __init__(self,sh,dv,args):
        super(encoder_mix,self).__init__()

        layers_dict = get_network(args.enc_layers)
        for l in layers_dict:
            if 'mu' in l['name'] or 'var' in l['name']:
                if 'conv' in l['name']:
                    l['num_features']=args.sdim*args.n_mix
                else:
                    l['num_units']=args.sdim*args.n_mix
            elif 'pi' in l['name']:
                l['num_units'] =args.n_mix

        #self.model = network.network()
        self.model=network.initialize_model(args, sh, args.enc_layers, dv, layers_dict=layers_dict)
        self.final_shape=np.array(self.model.temp.output_shape[1:], dtype=int)
        self.final_shape[0]/=args.n_mix


    def forward(self,inputs):

        # Run the predesigned network could be just the input
        self.model.temp.everything = True
        h,h1=self.model(inputs)
        self.model.temp.everything = False
        s_mu = h1['dense_mu'] if 'dense_mu' in h1 else h1['conv_mu']
        s_logvar= h1['dense_var'] if 'dense_var' in h1 else h1 ['conv_var']
        pi = torch.softmax(h1['dense_pi'], dim=1)

        return s_mu, s_logvar, pi,[h,h1]


class decoder_mix(nn.Module):
    def __init__(self,u_dim, z_dim,trans_shape, final_shape, dv, args):
        super(decoder_mix,self).__init__()
        self.u_dim=u_dim
        self.z_dim=z_dim
        if u_dim>0:
            self.dec_trans_top=nn.ModuleList([network.initialize_model(args, trans_shape, args.dec_trans_top, dv) for i in range(args.n_mix)])

        self.dec_conv_top=nn.ModuleList([network.initialize_model(args, final_shape,args.dec_layers_top, dv)for i in range(args.n_mix) ])
        f_shape=np.array(self.dec_conv_top[0].temp.output_shape)[1:]
        self.dec_conv_bot=network.initialize_model(args, f_shape, args.dec_layers_bot, dv)



    def forward(self,inputs,rng=None):

        xx=[]
        uu=[]
        for i,inp in enumerate(inputs):

            if self.u_dim>0:
                u = inp.narrow(len(inp.shape) - 1, 0, self.u_dim)
                z = inp.narrow(len(inp.shape) - 1, self.u_dim, self.z_dim)
                uu+=[self.dec_trans_top[i](u)[0]]
            else:
                z=inp
            x=self.dec_conv_top[i](z)[0]
            x=self.dec_conv_bot(x)[0]
            xx+=[x]

        xx = torch.stack(xx, dim=0)
        xx = torch.sigmoid(xx)
        if self.u_dim>0:
            uu = torch.stack(uu,dim=0)
        return xx, uu









