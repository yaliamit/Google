import numpy as np
import os
import time
from images import erode,make_images, make_sample
import torch



def test_models(ARGS, SMS, test, models, fout):

    len_test = len(test[0]);
    testMU = None;
    testLOGVAR = None;
    testPI = None
    CL_RATE = [];
    ls = len(SMS)
    CF=None
    #CF = [conf] + list(np.zeros(ls - 1))
    # Combine output from a number of existing models. Only hard ones move to next model?
    tes=[test[0],test[0],test[1]]
    if (ARGS[0].n_class):
        for sm, model, args, cf in zip(SMS, models, ARGS, CF):
            model.load_state_dict(sm['model.state.dict'])
            if 'vae' in args.type:
                testMU, testLOGVAR, testPI = model.initialize_mus(tes[0], args.OPT)
            print(cf)
            iid, RY, cl_rate, acc = model.run_epoch_classify(tes, 'test', fout=fout, num_mu_iter=args.nti, conf_thresh=cf)
            CL_RATE += [cl_rate]
            len_conf = len_test - np.sum(iid)
            print("current number", len_conf)
            if (len_conf > 0):
                print(float(cl_rate) / len_conf)
        print(np.float(np.sum(np.array(CL_RATE))) / len_test)
    else:
        for sm, model, args in zip(SMS, models, ARGS):
            model.load_state_dict(sm['model.state.dict'])
            if 'vae' in args.type:
                testMU, testLOGVAR, testPI = model.initialize_mus(tes[0], args.OPT)
            model.run_epoch(tes, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)



def train_model(model, args, ex_file, DATA, fout):

    if 'Linux' in os.uname():
        predir = '/ME/My Drive/'
    else:
        predir = '/Users/amit/Google Drive/'

    datadirs = predir + 'Colab Notebooks/STVAE/'


    fout.write("Num train:{0}\n".format(DATA[0][0].shape[0]))
    train=DATA[0]; val=DATA[1]; test=DATA[2]
    trainMU=None; trainLOGVAR=None; trPI=None
    valMU=None; valLOGVAR=None; valPI=None
    model.optimizer.param_groups[0]['lr']=args.lr
    model.get_scheduler(args)
    if 'vae' in args.type:
        trainMU, trainLOGVAR, trPI = model.initialize_mus(train[0], args.OPT)
        valMU, valLOGVAR, valPI = model.initialize_mus(val[0], args.OPT)


    VAL_ACC=[]
    tes = [test[0], test[0], test[1]]
    if (val[0] is not None):
        vall=[val[0],val[0],val[1]]
    tran = [train[0], train[0], train[1]]
    for epoch in range(args.nepoch):
        #print('time step',model.optimizer.param_groups[0]['lr'])
        #if (model.scheduler is not None):
        #    model.scheduler.step()
        t1 = time.time()
        if args.erode:
            tre = erode(args.erode, train[0])
            tran = [train[0], tre, train[1]]
        trainMU, trainLOGVAR, trPI, tr_acc = model.run_epoch(tran, epoch, args.num_mu_iter, trainMU, trainLOGVAR, trPI,d_type='train', fout=fout)
        if (val[0] is not None): # and (np.mod(epoch, 10) == 9 or epoch == 0)):
             _,_,_,val_acc=model.run_epoch(vall, epoch, args.nvi, valMU, valLOGVAR, valPI, d_type='val', fout=fout)
             VAL_ACC+=[val_acc[0],tr_acc[1]]
        else:
            VAL_ACC+=[tr_acc[0],tr_acc[1]]
        fout.write('{0:5.3f}s'.format(time.time() - t1))
        fout.flush()

    test_acc=np.zeros(2)
    if 'vae' in args.type:
        if (args.n_class):
            model.run_epoch_classify(tran, 'train', fout=fout, num_mu_iter=args.nti)
            model.run_epoch_classify(tes, 'test', fout=fout, num_mu_iter=args.nti)
        elif args.cl is None:
            LLG=model.compute_likelihood(test[0],250)
            rho=model.rho.detach().cpu().numpy()
            print('LLG',LLG,'rho',np.exp(rho)/np.sum(np.exp(rho)),file=fout)
            if args.hid_layers is None:
                testMU, testLOGVAR, testPI = model.initialize_mus(test[0], args.OPT)
                print('args.nti',args.nti,args.mu_lr)
                model.run_epoch(tes, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)
        fout.write('writing to ' + ex_file + '\n')
        args.fout=None
        make_sample(model,args, ex_file, datadirs=datadirs)
        make_images(train, model, ex_file, args, datadirs=datadirs)

    else:
        _,_,_,test_acc=model.run_epoch(tes, 0, args.nti, None, None, None, d_type='test', fout=fout)

    model.results=[np.array(VAL_ACC).transpose().reshape(-1,2)]+[test_acc[0]]
    return(model)