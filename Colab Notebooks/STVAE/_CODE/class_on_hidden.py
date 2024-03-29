
import numpy as np
import time
import network
from network_aux import initialize_model
import prep
from data import get_data_pre, get_pre
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.mixture import GaussianMixture
from images import create_image
import pickle
import os
from make import save_net_int
from data import DL

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
                INP = INP[0:args.hid_num_train]
            RR = []
            HVARS=[]
            for j in np.arange(0, INP.shape[0], 500):
                inp = INP[j:j + 500]
                rr, h_vars, losses, out_enc, _ = model.recon(inp, args.nti)
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

    args.num_train = args.hid_num_train
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

    args.num_train = args.hid_num_train
    datn = args.hid_dataset if args.hid_dataset is not None else args.dataset
    print('getting:' + datn,file=fout)
    DATA = get_data_pre(args, datn, device)
    args.num_class = DATA[0].num_class

    if 'ae' in args.type:
        _,[tr,tv,te]=prepare_recons(model,DATA,args,fout)
    elif args.embedd_type is not None:
        if args.optimizer_type=='LG' or args.hid_lr<0:
            res=train_LG(args,model,DATA)
        else:
            res=train_new_new(args, model, DATA, fout, device)
    else:
        return
    if hasattr(model, 'results'):
            model.results[1]=res[0]
    else:
            model.results=[None,res]
    return res


def train_LG(args,model,DATA):

        print('Using Logistic regression')
        train, test = embedd(DATA, model, args,to_dl=False)
        t1=time.time()
        lg=LogisticRegression(fit_intercept=True, solver='sag',multi_class='multinomial',max_iter=400,verbose=1, intercept_scaling=1., C=10.,penalty='l2')
        lg.fit(train[0], train[1])
        yh = lg.predict(train[0])
        print("Finished training:",time.time()-t1)
        print("train classification", np.mean(yh==train[1]))
        yh = lg.predict(test[0])
        res=np.mean(yh==test[1])
        print("test classification", res)

        return res

def embedd(DATA,model,args, to_dl=True):


    trdl = network.get_embedding(model,args,DATA[0])
    if args.AVG is not None:
        HW = (np.int32(trdl[0].shape[2] / args.AVG), np.int32(trdl[0].shape[3] / args.AVG))
        tra = torch.nn.functional.avg_pool2d(torch.from_numpy(trdl[0]), HW, HW)
        trdl = [tra, trdl[1]]
    trdl[0] = trdl[0].reshape(trdl[0].shape[0], -1)
    if to_dl:
        trdl = DL(list(zip(trdl[0], trdl[1])), batch_size=args.mb_size, num_class=args.num_class,
              num=trdl[0].shape[0], shape=trdl[0].shape[1:], shuffle=True)
    tedl = network.get_embedding(model,args,DATA[2])
    if args.AVG is not None:
        HW = (np.int32(tedl[0].shape[2] / args.AVG), np.int32(tedl[0].shape[3] / args.AVG))
        tea = torch.nn.functional.avg_pool2d(torch.from_numpy(tedl[0]), HW, HW)
        tedl = [tea, tedl[1]]
    tedl[0] = tedl[0].reshape(tedl[0].shape[0], -1)
    if to_dl:
        tedl = DL(list(zip(tedl[0], tedl[1])), batch_size=args.mb_size, num_class=args.num_class,
              num=tedl[0].shape[0], shape=tedl[0].shape[1:], shuffle=False)

    return trdl, tedl

def train_new_new(args,model,DATA,fout,device,net=None):

    print('In train new:')
    predir = get_pre()
    print(str(args))
    val = None
    trdl, tedl = embedd(DATA,model,args)
    if net is None:
        args.lr = args.hid_lr
        args.embedd=False
        args.embedd_type = None
        args.patch_size = None
        args.update_layers = None
        args.perturb = 0
        args.sched=args.hid_sched
        net=initialize_model(args, trdl.shape, args.hid_layers, device)
        network.get_scheduler(args,net.temp.optimizer)
        if args.hid_model:

            datadirs = predir + 'Colab Notebooks/STVAE/'
            sm = torch.load(datadirs + '_output/' + args.model[0]+'_classify' + '.pt', map_location='cpu')
            net.load_state_dict(sm['model.state.dict'])

    freq = 1
    freq_test=1
    t1 = time.time()
    if 'ga' in get_pre() and args.use_multiple_gpus is not None:
         print('loading on both gpus')
         net=torch.nn.DataParallel(net, device_ids=list(range(args.use_multiple_gpus)))
    for epoch in range(args.hid_nepoch):

        network.run_epoch(net, args, trdl, epoch, d_type='train',fout=fout,freq=freq)

        if (val is not None and np.mod(epoch, freq) == 0):
            network.run_epoch(net, args, val, epoch, d_type='val',fout=fout, freq=freq)

        if (freq - np.mod(epoch, freq) == 1):
            fout.write('epoch: {0} in {1:5.3f} seconds, LR {2:0.5f}\n'.format(epoch, time.time() - t1,
                                                                              net.temp.optimizer.param_groups[0]['lr']))
            fout.flush()
            t1 = time.time()
            if args.randomize:
                trdl, tedl = embedd(DATA, model, args)

        if (freq_test-np.mod(epoch,freq_test)==1):
            res = network.run_epoch(net, args, tedl, epoch, d_type='test_hidden',fout=fout, freq=freq)
        if hasattr(net,'scheduler') and net.scheduler is not None:
            net.scheduler.step()

    network.run_epoch(net, args, trdl, 0, d_type='test_tr',fout=fout, freq=1)

    res = network.run_epoch(net, args, tedl, 0, d_type='Final_test_hidden',fout=fout, freq=1)

    save_net_int(net, args.model_out+'_classify', args, predir)

    fout.flush()

    return res

def train_new_old(args,train,test,fout,device,net=None):

    #fout=sys.stdout
    print("In from hidden number of training",train.num)
    print('In train new:')
    print(str(args))
    predir = get_pre()
    val = None
    if net is None:
        args.lr = args.hid_lr
        args.embedd_type=None
        args.patch_size=None
        args.perturb=0
        #args.sched=[0,0]
        net=initialize_model(args, train.shape, args.hid_layers, device)
        network.get_scheduler(args)
        if args.hid_model:

            datadirs = predir + 'Colab Notebooks/STVAE/'
            sm = torch.load(datadirs + '_output/' + args.model[0]+'_classify' + '.pt', map_location='cpu')
            net.load_state_dict(sm['model.state.dict'])

    freq=10
    for epoch in range(args.hid_nepoch):
        if np.mod(epoch,freq)==0:
            t1=time.time()
        network.run_epoch(net,args,train,epoch, d_type='train',fout=fout, freq=freq)
        if (val is not None and np.mod(epoch,freq)==0):
                network.run_epoch(net,args,val,epoch, type='val',fout=fout, freq=freq)
        if (freq-np.mod(epoch,freq)==1):
            fout.write('epoch: {0} in {1:5.3f} seconds, LR {2:0.5f}\n'.format(epoch,time.time()-t1,args.temp.optimizer.param_groups[0]['lr']))
            fout.flush()
        #if hasattr(net,'scheduler') and net.scheduler is not None:
        #    net.scheduler.step()

    network.run_epoch(net,args,train, 0, d_type='test', fout=fout)

    res=network.run_epoch(net,args,test, 0, d_type='test', fout=fout)

    save_net_int(net, args.model_out + '_classify', args, predir)

    fout.flush()

    return res




