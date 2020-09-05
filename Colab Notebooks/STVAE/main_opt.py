import sys
import os
from main import main_loc
import matplotlib.pyplot as plt
from mprep import get_data_pre
import argparse
import aux as aux
import pylab as py
import torch
import mprep as mprep
import numpy as np
from aux_colab import copy_to_content, seq, train_net, run_net
from Conv_data import get_pre

predir=get_pre
# if 'Linux' in os.uname():
#     from google.colab import drive
#     drive.mount('/ME')
#     predir='/ME/My Drive/'
# else:
#     predir='/Users/amit/Google Drive/'


datadirs=predir+'Colab Notebooks/STVAE/'
sys.path.insert(1, datadirs)
sys.path.insert(1, datadirs+'_CODE')



def test_aug(aug='aff'):


    net = main_loc('_pars/pars_aug')

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='Variational Autoencoder with Spatial Transformation')
    parser = aux.process_args(parser)
    par_file = '_pars/pars_aug'
    f = open(par_file + '.txt', 'r')
    args = parser.parse_args(f.read().split())
    f.close()
    args.datadirs = datadirs

    DATA = get_data_pre(args, args.dataset)
    net.trans = aug
    ims = DATA[0][0].transpose(0, 2, 3, 1)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ims_def = net.deform_data(torch.from_numpy(DATA[0][0][0:net.bsz]).to(device)).detach().cpu().numpy().transpose(0, 2, 3,
                                                                                                               1)

    print(net.trans)

    return ims[0:net.bsz], ims_def, net

def test_loss():

    ims, ims_def, net=test_aug()

    ims=torch.from_numpy(ims.transpose(0,3,1,2))
    ims_def=torch.from_numpy(ims_def.transpose(0,3,1,2))
    out,_=net.forward(ims)
    outa,_=net.forward(ims_def)

    loss1=net.get_embedd_loss(out,outa)
    loss2=net.get_embedd_loss_a(out,outa)

    print(loss1,loss2)

    exit()


#test_loss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device.type=='cpu')
#copy_to_content('try',predir)
#net=run_net('t_par', device)

#plt.plot(net.results[0][0])
#plt.show()

#print("helo")
#copy_to_content('pars_emb_cifar',predir)
#os.system("echo --layerwise >> pars_big_cl_a.txt")
run_net('t_par', device)
#exit()
#seq('pars_emb_mnist',predir, device)
#np.random.seed(123456)

# ims, ims_def, _=test_aug()
# kk=np.random.randint(0,500,10)
#
# fig=py.figure(figsize=(2,10))
# for i,k in enumerate(kk):
#     py.subplot(10,2,2*i+1)
#     py.imshow(ims[k])
#     py.axis('off')
#     py.subplot(10,2,2*i+2)
#     py.imshow(ims_def[k])
#     py.axis('off')
# py.show()

print("hello")


