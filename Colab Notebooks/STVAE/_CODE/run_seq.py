import os

name='pars_emb_AE'
for j in range(5):
    nm=name+'str(j+1)'
    os.system('python main_opt.py 1 '+nm)