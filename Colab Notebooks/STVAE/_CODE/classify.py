import torch
from mix import STVAE_mix
import numpy as np
import time
from data import get_pre, DL
import os

def classify_by_likelihood(args,model,DATA, device, fout):
    run_classify(args, DATA[0], model, device, fout, 'train')
    run_classify(args, DATA[2], model,  device, fout, 'test')
    fout.write('DONE\n')


def run_classify(args,train,model,device, fout, type):
    VV=[]
    datadir=os.path.join(get_pre(),'Colab Notebooks','STVAE','_output')
    y=[]

    Dtr=[]
    i=0
    for t in train:
        print(i)
        i+=1
        Dtr+=[t[0][0].numpy()]
        y+=[t[0][1].numpy()]
    Dtr=np.concatenate(Dtr)
    y=np.concatenate(y)
    tr=DL(list(zip(Dtr,y)),batch_size=args.mb_size, num_class=args.classify,
             num=Dtr.shape[0], shape=Dtr.shape[1:],shuffle=False)
    for cl in range(args.classify):
        t1 = time.time()
        fout.write(str(cl)+'\n')
        fout.flush()
        ex_file = args.model_out+'_'+str(cl)
        model.load_state_dict(torch.load(os.path.join(datadir,ex_file + '.pt'), map_location=device)['model.state.dict'])
        V=run_epoch_classify(args, model,tr,device, args.nti,fout)
        VV+=[V.cpu()]
        fout.write('classify: {0} in {1:5.3f} seconds\n'.format(cl,time.time()-t1))

    VVV=np.stack(VV,axis=1)
    hy=np.argmin(VVV,axis=1)

    acc=np.mean(np.equal(hy,y))
    fout.write('====> {} Accuracy {:.4F}\n'.format(type,acc))


def run_epoch_classify(args, model, train, device, num_mu_iter,fout):


        sdim=model.s_dim
        #mu, logvar, pi = model.initialize_mus(train[0], sdim, True)

        model.eval()
        like=[]

        tra = iter(train)
        for j in np.arange(0, train.num, train.batch_size):
                BB, indlist = next(tra)
                data_in = BB[0].to(device)
                _, _, [recloss, totloss], _, _, _ = model.recon(args, data_in,num_mu_iter=args.nti)
                like+=[recloss+totloss]
        like=torch.cat(like)
        return like



