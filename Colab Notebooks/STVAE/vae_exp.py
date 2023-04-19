import os

sdims=['5','10','20']
n_mixs=['1','2']
files=['pars_tvae_orig','pars_vae_mnist']
num_trains=['1000'] #,'10000','50000']

for file in files:
  for sdim in sdims:
    for n_mix in n_mixs:
        for num_train in num_trains:
                argss=file+' --sdim='+sdim+' --n_mix='+n_mix+' --num_train='+num_train
                with open('ACC','a') as f:
                    f.write(argss+'\n')
                comm='python3 main_opt.py 0 '+argss +' --n_class=10 --by_class > junk'
                os.system(comm)
                comm='python3 main_opt.py 0 '+argss +' --n_class=1 --classify=10 | grep Accuracy >> ACC'
                os.system(comm)