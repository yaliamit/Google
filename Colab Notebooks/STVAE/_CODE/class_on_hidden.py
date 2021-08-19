
import numpy as np
import time
import network
import prep
from data import get_data_pre
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture
from images import create_image
import pickle
import os

def prepare_recons(model, DATA, args,fout):
    dat = []
    HV=[]
    tips=['train','val','test']
    rr=range(0,3)

    for k in rr:
        totloss = 0
        recloss = 0
        if (DATA[k][0] is not None):
            INP = torch.from_numpy(DATA[k][0])
            if k==0:
                INP = INP[0:args.network_num_train]
            RR = []
            HVARS=[]
            for j in np.arange(0, INP.shape[0], 500):
                inp = INP[j:j + 500]
                rr, h_vars, losses, out_enc= model.recon(inp, args.nti)
                recloss+=losses[0]
                totloss+=losses[1]
                RR += [rr.detach().cpu().numpy()]
                HVARS += [h_vars.detach().cpu().numpy()]
            RR = np.concatenate(RR)
            HVARS = np.concatenate(HVARS)
            tr = RR.reshape(-1, model.input_channels,model.h,model.w)
            dat += [tr]
            HV+=[HVARS]
            if (k==2):
                fout.write('\n====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(tips[k],
                                                            0,recloss / INP.shape[0], (recloss+totloss)/INP.shape[0]))

            print(k,recloss/INP.shape[0],(totloss+recloss)/INP.shape[0])
        else:
            dat += [DATA[k]]
            HV += [DATA[k]]
    print("Hello")

    return dat, HV, out_enc

def cluster_hidden(model,args,device,data,datadirs,ex_file):

    args.num_train = args.network_num_train
    data[0]=[data[0][0][0:args.num_train],data[0][1][0:args.num_train]]
    exa_file = datadirs + '_output/' + ex_file.split('.')[0]
    if os.path.isfile(exa_file+'.npz'):
        with open(exa_file+'.npz','rb') as f:
            A=np.load(f)
            tr=A['arr_0']
            te=A['arr_1']
    else:
        _, [tr, tv, te] = prepare_recons(model, data, args, args.fout)
        tr=tr[:,0:model.s_dim]
        te=te[:,0:model.s_dim]
        exa_file = datadirs + '_output/' + ex_file.split('.')[0]
        with open(exa_file+'.npz','wb') as f:
            np.savez(f,tr,te)

    gm = GaussianMixture(n_components=args.hidden_clusters,covariance_type='full',tol=.1).fit(tr)
    print("training score",gm.score(tr))
    print("testing score",gm.score(te))
    s=gm.sample(args.num_sample)
    print('Finished training mixture model',gm.weights_)
    S=[torch.from_numpy(s[0]).float().to(device),torch.from_numpy(s[1]).float().to(device)]
    s=S[0].reshape(-1,1,model.s_dim)
    s = s.transpose(0, 1)
    with torch.no_grad():
        model.decoder_m.cluster_hidden=True
        xx = model.decoder_and_trans(s, train=False)
    #xx=model.decoder_m.forward_specific(S,model.enc_conv)


    X=xx.detach().cpu().numpy().squeeze()
    exa_file = datadirs + '_Images/' + ex_file.split('.')[0]
    with open(exa_file+'_cl'+str(args.hidden_clusters)+'_gm.pickle', 'wb') as f:
        pickle.dump(gm,f)
    np.random.shuffle(X)
    create_image(X[0:100], model, exa_file)
    X=np.uint8(X*255)
    ex_file=datadirs+'_Samples/'+ex_file.split('.')[0]+'.npy'

    np.save(ex_file,X)


def pre_train_new(model,args,device,fout, data=None):
    if args.hid_layers is None:
        return

    args.num_train = args.network_num_train
    datn = args.hid_dataset if args.hid_dataset is not None else args.dataset
    print('getting:' + datn)
    DATA = get_data_pre(args, datn)
    args.num_class = np.int(np.max(DATA[0][1]) + 1)

    if 'ae' in args.type:
        _,[tr,tv,te]=prepare_recons(model,DATA,args,fout)
    elif args.embedd:
        tr = model.get_embedding(DATA[0][0][0:args.network_num_train]) #.detach().cpu().numpy()
        tr = tr.reshape(tr.shape[0], -1)
        te = model.get_embedding(DATA[2][0]) #.detach().cpu().numpy()
        te = te.reshape(te.shape[0], -1)
    else:
        return

    trh = [tr, DATA[0][1][0:args.network_num_train]]
    teh = [te, DATA[2][1]]

    args.embedd = False
    args.update_layers=None
    args.lr=args.hid_lr
    if 'xla' in device.type:
        return [trh, teh]
    else:
        res=train_new(args,trh,teh,fout,device)
        if hasattr(model, 'results'):
            model.results[1]=res[0]
        else:
            model.results=[None,res]
        return res


def train_new(args,train,test,fout,device):
   
    if args.optimizer=='LG' or args.hid_lr<0:
        print('Using Logistic regression')
        t1=time.time()
        lg=LogisticRegression(fit_intercept=True, solver='lbfgs',multi_class='multinomial',max_iter=4000,verbose=1, intercept_scaling=1., C=10.,penalty='l2')
        lg.fit(train[0], train[1])
        yh = lg.predict(train[0])
        print("Finished training:",time.time()-t1)
        print("train classification", np.mean(yh==train[1]))
        yh = lg.predict(test[0])
        res=np.mean(yh==test[1])
        print("test classification", res)


    else:
        res=train_new_old(args, train, test, fout, device)

    return res

def train_new_old(args,train,test,fout,device,net=None):

    #fout=sys.stdout
    print("In from hidden number of training",train[0].shape[0])
    print('In train new:')
    print(str(args))
    val = None
    if net is None:
        args.lr = args.hid_lr
        args.hid_lnti, args.hid_layers_dict = prep.get_network(args.hid_layers)
        args.perturb=0
        #args.sched=[0,0]
        net=network.network(device,args,args.hid_layers_dict, args.hid_lnti, sh=train[0].shape).to(device)

        net.get_scheduler(args)

    tran=[train[0],train[0],train[1]]
    for epoch in range(args.hid_nepoch):

        t1=time.time()
        net.run_epoch(tran,epoch, d_type='train',fout=fout)
        if (val is not None):
                net.run_epoch(val,epoch, type='val',fout=fout)
        if (np.mod(epoch,10)==9 or epoch==0):
            fout.write('epoch: {0} in {1:5.3f} seconds, LR {2:0.5f}'.format(epoch,time.time()-t1,net.optimizer.param_groups[0]['lr']))
            fout.flush()
        if hasattr(net,'scheduler') and net.scheduler is not None:
            net.scheduler.step()

    tes=[test[0],test[0],test[1]]
    _,_,_,res=net.run_epoch(tes, 0, d_type='test', fout=fout)

    fout.flush()

    return res




