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

predir=get_pre()
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

#copy_to_content('pars_tvae_orig',predir)
net=run_net('pars_tvae_orig', device)
#save_net(net,'pars_mnist_a',predir)

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


