
import sys
import numpy as np
import time
import network
import torch
import mprep
import models_images
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LinearRegression, Ridge

def pre_train_new(model,args,device,fout, data=None):
    if args.hid_layers is None:
        return

    args.num_train = args.network_num_train
    if args.embedd:
        datn = args.hid_dataset if args.hid_dataset is not None else args.dataset
        print('getting:' + datn)
        DATA = mprep.get_data_pre(args, datn)
        args.num_class = np.int(np.max(DATA[0][1]) + 1)
        tr = model.get_embedding(DATA[0][0][0:args.network_num_train]) #.detach().cpu().numpy()
        tr = tr.reshape(tr.shape[0], -1)
        trh = [tr, DATA[0][1][0:args.network_num_train]]

        te = model.get_embedding(DATA[2][0]) #.detach().cpu().numpy()
        te = te.reshape(te.shape[0], -1)
        teh = [te, DATA[2][1]]
    else:
        dat, DATA = models_images.prepare_recons(model, data, args, fout)
        trh=[DATA[0][0],DATA[0][1]]
        teh=[DATA[2][0],DATA[2][1]]
    args.embedd = False
    args.update_layers=None
    args.lr=args.hid_lr
    res=train_new(args,trh,teh,fout,device)
    model.results[1]=res[0]
    return res


def train_new(args,train,test,fout,device):
   
    if args.optimizer=='LG':
        print('Using Logistic regression')
        lg=LogisticRegression(fit_intercept=True, solver='lbfgs',multi_class='multinomial',max_iter=1000, intercept_scaling=1, C=.1,penalty='l2')
        lg.fit(train[0], train[1])
        yh = lg.predict(train[0])
        print("train classification", np.mean(yh==train[1]))
        yh = lg.predict(test[0])
        print("test classification", np.mean(yh==test[1]))
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
    args.hid_lnti, args.hid_layers_dict = mprep.get_network(args.hid_layers)
    #net=network.network(device,args,args.hid_layers_dict, args.hid_lnti, fout=fout, sh=train[0].shape).to(device)
    net=network.network(device,args,args.hid_layers_dict, args.hid_lnti, sh=train[0].shape).to(device)
    #temp = torch.zeros([1] + list(train[0].shape[1:])).to(device)
    # Run the network once on dummy data to get the correct dimensions.
    #bb = net.forward(temp)
    scheduler=None
    tran=[train[0],train[0],train[1]]
    for epoch in range(args.hid_nepoch):
        if (scheduler is not None):
            scheduler.step()
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




