import sys
import os
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



os.environ['KMP_DUPLICATE_LIB_OK']='True'

# if 'Linux' in os.uname():
#     from google.colab import drive
#     drive.mount('/ME')
#     predir='/ME/My Drive/'
# else:
#     predir='/Users/amit/Google Drive/'


datadirs=predir+'Colab Notebooks/STVAE/'
sys.path.insert(1, datadirs)
sys.path.insert(1, datadirs+'_CODE')

# aa=torch.round(torch.rand(3,7,8,8)*10)
# ss=shifts([1,1])
# bb=ss(aa)



def  to_npy():




    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='Variational Autoencoder with Spatial Transformation')
    parser = aux.process_args(parser)
    par_file = 'pars_emb_cifar'
    f = open(par_file + '.txt', 'r')
    args = parser.parse_args(f.read().split())
    f.close()
    args.datadirs = datadirs

    DATA = get_data_pre(args, args.dataset)
    ims = np.uint8(DATA[0][0][0:10000].transpose(0, 2, 3, 1)*255.)
    np.save(predir+'/LSDA_data/CIFAR/cifar10.npy',ims)




#test_loss()
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

def resnet_try():
    from torchvision import models
    resnet = models.resnet18(pretrained=True)
    resnet.eval()
    from PIL import Image
    img_cat = Image.open("/Users/amit/Desktop/yali_yann.jpg").convert('RGB')
    #

    from torchvision import transforms
    #
    # Create a preprocessing pipeline
    #
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

    img_cat_preprocessed = preprocess(img_cat)
    #
    batch_img_cat_tensor = torch.unsqueeze(img_cat_preprocessed, 0)

    out = resnet(batch_img_cat_tensor)

#to_npy()
#copy_to_content('pars_tvae_conv',predir)

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

os.system('cat '+par_file+'.txt junk>'+par_file+'_temp.txt')



temp_file=par_file+'_temp'
if count_non<4:
    net,_,args=run_net(temp_file, device)
    # if net.optimizer_type=='Adam':
    #     net.optimizer = torch.optim.Adam(net.optimizer.param_groups[0]['params'], lr=net.lr, weight_decay=net.wd)
    # else:
    #     net.optimizer = torch.optim.SGD(net.optimizer.param_groups[0]['params'], lr=net.lr)
    #
    # net,_,args=run_net(temp_file, device, net)
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


