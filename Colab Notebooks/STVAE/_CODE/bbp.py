
import torch
import os
import prep
import network

os.environ['KMP_DUPLICATE_LIB_OK']='True'


ww=torch.load('/home/amit/ga/Google/Colab Notebooks/ssl-playground-master/backbone_weights.pth')


for s,t in ww['bb'].items():
    print(s)
    print(t)
ww['bb'].update(ww['pp'])


par_file='pars_emb_direct_temp'
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
torch.save({'args': arg,'model.state.dict': model.state_dict()}, '/home/amit/ga/Google/Colab Notebooks/STVAE/_output/dd_new.pt')
