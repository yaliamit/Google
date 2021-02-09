import torch
from layers import attention
import os
import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
os.environ['KMP_DUPLICATE_LIB_OK']='True'

b=5
l=10
d=7

zz=torch.ones(l,l)
for i,z in enumerate(zz):
    z[i+1:]=0

np_mask = np.triu(np.ones((1,l, l)),
    k=1).astype('uint8')
np_mask=torch.from_numpy(np_mask) == 0
v=torch.rand(5,10,7)

LL=nn.Linear(d,d)

vl=LL(v)

attention(v,v,v,d,np_mask)

