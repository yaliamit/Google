
import torch
import os
import prep
import network

os.environ['KMP_DUPLICATE_LIB_OK']='True'


ww=torch.load('/ga/Googe/Colab\ Notebooks/ssl-playground-master/backbone_weights.pth')
ww['bb'].update(ww['pp'])

par_file='pars_emb_direct'
arg = prep.setups(par_file)
sh=[3,32,32]
device='cpu'
model=network.initialize_model(arg,sh,arg.layers,device)
params = model.named_parameters()
dict_params = dict(params)


for a,b in zip(dict_params.items(),ww['bb'].items()):
    a[1].data.copy_(b[1])


for n,p in dict_params.items():
    print(n)

arg.fout=None
torch.save({'args': arg,'model.state.dict': model.state_dict()}, '/ga/Googe/Colab\ Notebooks/STVAE/_output/dd_new.pt')