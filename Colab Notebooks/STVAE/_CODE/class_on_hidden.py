
import numpy as np
import time
import network
import prep
from data import get_data_pre
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression, Ridge

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
                rr, h_vars, losses= model.recon(inp, args.nti)
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

    return dat, HV



def pre_train_new(model,args,device,fout, data=None):
    if args.hid_layers is None:
        return

    args.num_train = args.network_num_train
    datn = args.hid_dataset if args.hid_dataset is not None else args.dataset
    print('getting:' + datn)
    DATA = get_data_pre(args, datn)
    args.num_class = np.int(np.max(DATA[0][1]) + 1)

    if 'vae' in args.type:
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
    res=train_new(args,trh,teh,fout,device)
    if hasattr(model, 'results'):
        model.results[1]=res[0]
    else:
        model.results=[None,res]
    return res


def train_new(args,train,test,fout,device):
   
    if args.optimizer=='LG':
        print('Using Logistic regression')
        lg=LogisticRegression(fit_intercept=True, solver='lbfgs',multi_class='multinomial',max_iter=1000, intercept_scaling=1, C=.1,penalty='l2')
        lg.fit(train[0], train[1])
        yh = lg.predict(train[0])
        print("train classification", np.mean(yh==train[1]))
        yh = lg.predict(test[0])
        res=np.mean(yh==test[1])
        print("test classification", res)

    else:
        res=train_new_old(args, train, test, fout, device)

    return res

def train_new_old(args,train,test,fout,device):

    #fout=sys.stdout
    print("In from hidden number of training",train[0].shape[0])
    print('In train new:')
    print(str(args))
    val = None
    args.lr = args.hid_lr
    args.hid_lnti, args.hid_layers_dict = prep.get_network(args.hid_layers)
    args.perturb=0
    net=network.network(device,args,args.hid_layers_dict, args.hid_lnti, sh=train[0].shape).to(device)
    #temp = torch.zeros([1] + list(train[0].shape[1:])).to(device)
    # Run the network once on dummy data to get the correct dimensions.
    #bb = net.forward(temp)
    net.get_scheduler(args)

    tran=[train[0],train[0],train[1]]
    for epoch in range(args.hid_nepoch):
        if (net.scheduler is not None):
            net.scheduler.step()
        t1=time.time()
        net.run_epoch(tran,epoch, d_type='train',fout=fout)
        if (val is not None):
                net.run_epoch(val,epoch, type='val',fout=fout)
        if (np.mod(epoch,10)==9 or epoch==0):
            fout.write('epoch: {0} in {1:5.3f} seconds'.format(epoch,time.time()-t1))
            fout.flush()


    tes=[test[0],test[0],test[1]]
    _,_,_,res=net.run_epoch(tes, 0, d_type='test', fout=fout)

    fout.flush()

    return res




