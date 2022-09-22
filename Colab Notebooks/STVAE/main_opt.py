import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from main import main_loc
import matplotlib.pyplot as plt
from data import get_data_pre
import argparse
import aux as aux
import pylab as py
import torch
import prep as mprep
import numpy as np
from aux_colab import copy_to_content, seq, train_net, run_net, save_net
from data import get_pre
from layers import *

predir=get_pre()
import os





# if 'Linux' in os.uname():
#     from google.colab import drive
#     drive.mount('/ME')
#     predir='/ME/My Drive/'
# else:
#     predir='/Users/amit/Google Drive/'


datadirs=predir+'Colab Notebooks/STVAE/'
sys.path.insert(1, datadirs)
sys.path.insert(1, datadirs+'_CODE')

print(sys.argv)
if not torch.cuda.is_available():
    device=torch.device("cpu")
else:
    if len(sys.argv)==1:
        s="cuda:"+"0"
    else:
        s="cuda:"+sys.argv[1]
    device=torch.device(s)
print(device)

os.system('rm junk')
count_non=0
for a in sys.argv:
    if '--' in a:
        os.system('echo \"'+a+'\">> junk')
    else:
        count_non+=1


if count_non<3:
    par_file='t_par'
else:
    par_file=sys.argv[2]
    print(par_file)

os.system('grep -v "#" '+par_file+'.txt > junk1')
os.system('cat junk1 junk>'+par_file+'_temp.txt')



temp_file=par_file+'_temp'
if count_non<4:
    net,_,args=run_net(temp_file, device)
    if not args.run_existing:
        save_net(net,temp_file,predir)
else:
    tlay=None
    toldn=None
    if len(sys.argv)>4:
        tlay=sys.argv[4]
        toldn=sys.argv[5]
    seq(temp_file,predir, device, tlay=tlay, toldn=toldn)


os.system('rm junk')

print("hello")


