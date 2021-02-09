import torch
import torch.nn as nn
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MaskedConvolution(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, pd, mask=None):
        super(MaskedConvolution, self).__init__(in_channels, out_channels, kernel_size,padding=pd,bias=False)

        Cout, Cin, kh = self.weight.size()
        pre_mask = np.ones_like(self.weight.data.cpu().numpy()).astype(np.float32)
        yc = kh // 2


        if mask=='A':
            pre_mask[:,:,yc:]=0.0
        elif mask=='B':
            pre_mask[:,:,yc+1:]=0.0

        print(pre_mask[0, :, :])

        self.register_buffer("mask", torch.from_numpy(pre_mask))

    def __call__(self, x):
        self.weight.data = self.weight.data * self.mask
        return super(MaskedConvolution, self).forward(x)



# Dimension
d=2
# Sentence length
l=4
# Batchsize
b=1

# data
a=torch.rand(b,d,l)

CC=MaskedConvolution(d,d,3,1,mask='B')
# Just forcing certain values to check calculation
CC.weight.data=torch.ones_like(CC.weight.data)
CC.weight.data[0,1,0]=2
ac=CC(a)
print('a',a)
print('ac',ac)