import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_mix_by_class import STVAE_mix_by_class
import numpy as np
import os
import sys
import argparse
import time
from Conv_data import get_data
from models import  get_scheduler
import aux
from class_on_hidden import train_new
import network
import mprep
from get_net_text import get_network
from model_cluster_labels import assign_cluster_labels
from classify import classify





########################################
# Main Starts
#########################################
os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
    description='Variational Autoencoder with Spatial Transformation')


args=aux.process_args(parser)
ARGS, STRINGS, EX_FILES, SMS = mprep.get_names(args)
fout, device, DATA= mprep.setups(args, EX_FILES)
ARGS[0].binary_thresh=args.binary_thresh
sh=DATA[0][0].shape
# parse the existing network coded in ARGS[0]
arg=ARGS[0]
arg.lnti, arg.layers_dict = get_network(arg.layers)
model = network.network(device, arg, arg.layers_dict, arg.lnti).to(device)
temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(device)
bb = model.forward(temp)

sample=args.sample
classify=args.classify
reinit=args.reinit
run_existing=args.run_existing
conf=args.conf
embedd=args.embedd
num_test=args.num_test
num_train=args.num_train
nepoch=args.nepoch
lr=args.lr
network=args.network
ARGS[0].nti=args.nti
ARGS[0].num_test=num_test

fout.write(str(ARGS[0]) + '\n')
fout.flush()
# if (args.classify):
#     t1 = time.time()
#     classify(train,test,image_dim,opt_pre,opt_post,opt_mix,opt_class,device,args,fout,locals())
#     fout.write('Classified in {1:5.3f} seconds\n'.format(time.time()-t1))
#     exit()



model.load_state_dict(SMS[0]['model.state.dict'])
train=DATA[0]
test=DATA[2]
II=[]
for k in range(10):
    II+=[np.where(train[1]==k)[0]]
IIt=[]
for k in range(10):
    IIt+=[np.where(test[1]==k)[0]]


cc=model.get_binary_signature(train,lays=['pool1','pool2'])
cc=cc.numpy()
cc1=cc.reshape(-1)
print('all',np.mean(cc1),np.quantile(cc1,[.25,.5,.75]),np.max(cc1))
for k in range(10):
    cc1 = cc[II[k]][:,II[k]].reshape(-1)
    print(k, np.mean(cc1), np.quantile(cc1, [.25, .5, .75]),np.max(cc1))

# for k in range(10):
#     for l in range(k):
#         cc1 = cc[II[k]][:, II[l]].reshape(-1)
#         print(k,l, np.mean(cc1), np.quantile(cc1, [.25, .5, .75]),np.max(cc1))


cc=model.get_binary_signature(train,test,lays=['pool1','pool2'])
cc=cc.numpy()
cc1=cc.reshape(-1)
print('all',np.mean(cc1),np.quantile(cc1,[.25,.5,.75]),np.max(cc1))
for k in range(10):
    cc1 = cc[II[k]][:,IIt[k]].reshape(-1)
    print(k, np.mean(cc1), np.quantile(cc1, [.25, .5, .75]),np.max(cc1))

# for k in range(10):
#     for l in range(k):
#         cc1 = cc[II[k]][:, IIt[l]].reshape(-1)
#         print(k,l, np.mean(cc1), np.quantile(cc1, [.25, .5, .75]),np.max(cc1))

print("Hello")

