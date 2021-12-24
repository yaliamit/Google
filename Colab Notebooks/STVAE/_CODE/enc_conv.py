import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from get_net_text import get_network
import network
import sys



class ENC_DEC(nn.Module):
    def __init__(self, sh, device, args, layers):
        super(ENC_DEC, self).__init__()


        lnti,layers_dict = get_network(layers)
        self.model=network.network()
        network.initialize_model(self.model, args, sh, lnti, layers_dict,  device)
        print('done')


    def forw(self,input, args):

        self.model.temp.everything=True
        out,out1=self.model.forward(input, args)

        return(out,out1)

    def bkwd(self,input):
        out=self.model.backwards(input)
        return(out)


