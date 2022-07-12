import numpy as np
import os
import time
from images import erode,make_images, make_sample
import torch
from data import get_pre, DL
from network import get_scheduler, run_epoch
import copy
from mix import initialize_mus

def save_net_int(model,model_out,args,predir):

  print(model_out, file=args.fout)
  fout=args.fout
  args.fout=None
  args_temp=copy.deepcopy(args)


  if model_out is not None:
      ss=model_out+'.pt'
  else:
      ss='network.pt'

  if 'Users/amit' in predir:
      torch.save({'args': args_temp,
          'model.state.dict': model.state_dict()}, predir + 'Colab Notebooks/STVAE/_output/' + ss)
  else:
    torch.save({'args': args_temp,
        'model.state.dict': model.state_dict()}, predir+'Colab Notebooks/STVAE/_output/'+ss,_use_new_zipfile_serialization=False)

  args.fout=fout


def test_models(ARGS, SMS, test, models, fout):

    len_test=test.num
    print('len_test',test.num,'nti',ARGS[0].nti)
    testLOGVAR = None;
    testPI = None
    CL_RATE = [];
    ls = len(SMS)
    CF=None
    #CF = [conf] + list(np.zeros(ls - 1))
    # Combine output from a number of existing models. Only hard ones move to next model?
    if (ARGS[0].n_class):
        for sm, model, args, cf in zip(SMS, models, ARGS, CF):
            model.load_state_dict(sm['model.state.dict'])
            if 'vae' in args.type:
                testMU, testLOGVAR, testPI = model.initialize_mus(test.shape[0], model.final_shape, model.n_mix)
            print(cf)
            iid, RY, cl_rate, acc = model.run_epoch_classify(test, 'test', fout=fout, num_mu_iter=args.nti, conf_thresh=cf)
            CL_RATE += [cl_rate]
            len_conf = len_test - np.sum(iid)
            print("current number", len_conf)
            if (len_conf > 0):
                print(float(cl_rate) / len_conf)
        print(np.float(np.sum(np.array(CL_RATE))) / len_test)
    else:
        for sm, model, args in zip(SMS, models, ARGS):
            testMU = None;testLOGVAR = None;testPI = None
            #model.load_state_dict(sm['model.state.dict'])
            model.bsz=args.mb_size
            if 'vae' in args.type:
                testMU, testLOGVAR, testPI = model.initialize_mus(test.shape[0], model.final_shape, model.n_mix)
            model.run_epoch(test, 0, args.nti, testMU, testLOGVAR, testPI, d_type='test', fout=fout)



def train_model(model, args, ex_file, DATA, fout):


    predir=get_pre()
    datadirs = predir + 'Colab Notebooks/STVAE/'


    train=DATA[0]; val=DATA[1]; test=DATA[2]
    trainMU=None; trainLOGVAR=None; trPI=None
    valMU=None; valLOGVAR=None; valPI=None





    num_train= train.num if type(train) is DL else train[0].shape[0]
    num_test=0
    if test is not None:
        num_test= test.num if type(test) is DL else test[0].shape[0]

    if type(val) is DL:
        num_val = val.num
    #elif val is not None:
    #   num_val=val[0].shape[0]

    fout.write("Num train:{0}, Num test:{1}\n".format(num_train,num_test))

    if 'ae' in args.type:
        trainMU, trainLOGVAR, trPI = initialize_mus(num_train, model.final_shape, model.n_mix)
        if val is not None:
            valMU, valLOGVAR, valPI = initialize_mus(num_val,model.final_shape, model.n_mix)

    time1=time.time()
    VAL_ACC=[]


    if args.OPT and args.cont_training:
        print("Updating training optimal parameters before continuing")
        trainMU, trainLOGVAR, trPI, tr_acc = model.run_epoch(args,train, 0, args.nti, trainMU, trainLOGVAR, trPI,
                                                             d_type='test', fout=fout)

    if 'ga' in get_pre() and args.use_multiple_gpus is not None:
         print('loading on both gpus')
         model=torch.nn.DataParallel(model, device_ids=list(range(args.use_multiple_gpus)))
         #args.temp.loss=torch.nn.DataParallel(args.temp.loss, device_ids=list(range(args.use_multiple_gpus)))
         optimizer=model.module.temp.optimizer

    else:
        optimizer=model.temp.optimizer
    #model=model.to(args.temp.dv)
    #model.temp.loss=model.temp.loss.to(model.temp.dv)
    scheduler=get_scheduler(args,optimizer)
    optimizer.param_groups[0]['lr']=args.lr

    for epoch in range(args.nepoch):

        t1 = time.time()
        if 'ae' not in args.type:
            tr_acc = run_epoch(model,args,train, epoch, d_type='train', fout=fout)
        else:
            trainMU, trainLOGVAR, trPI, tr_acc = model.run_epoch(args,train, epoch, args.num_mu_iter, trainMU, trainLOGVAR, trPI,d_type='train', fout=fout)

        if (val is not None):
            if 'ae' not in args.type:
                val_acc=run_epoch(model,args,val, epoch, d_type='val', fout=fout)
            else:
                _,_,_,val_acc=model.run_epoch(args, val, epoch, args.nvi, valMU, valLOGVAR, valPI, d_type='val', fout=fout)
                VAL_ACC+=[val_acc[0],tr_acc[1]]
        else:
            VAL_ACC+=[tr_acc[0],tr_acc[1]]
        time2=time.time()
        fout.write('Time {0:5.3f}s, LR {1:f}'.format(time2 - t1,optimizer.param_groups[0]['lr']))

        fout.flush()
        time2=time.time()
        if time2-time1>600 or epoch % 100==0:
            save_net_int(model, args.model_out+'_'+str((epoch)), args, predir)
            time1=time2
        if scheduler is not None:
            scheduler.step()
    test_acc=np.zeros(2)

    if 'ae' in args.type:
        make_images(test, model, ex_file, args, datadirs=datadirs)
        #make_sample(model,args, ex_file, datadirs=datadirs)

        if (args.n_class):
            #model.run_epoch_classify(train, 'train', fout=fout, num_mu_iter=args.nti)
            #model.run_epoch_classify(test, 'test', fout=fout, num_mu_iter=args.nti)
            pass
        elif args.cl is None:
            #if not args.OPT:
                #LLG=model.compute_likelihood(test[0],250)
                #print('LLG', LLG, file=fout)
            rho=model.rho.detach().cpu().numpy()
            print('rho',np.exp(rho)/np.sum(np.exp(rho)),file=fout)
            if args.hid_layers is None:
                testMU, testLOGVAR, testPI = initialize_mus(num_train, model.final_shape, model.n_mix)
                print('args.nti',args.nti,args.mu_lr,file=fout)
                model.run_epoch(args, train,  0, args.nti, testMU, testLOGVAR, testPI, d_type='train_test', fout=fout)
                testMU, testLOGVAR, testPI = initialize_mus(num_test, model.final_shape, model.n_mix)
                model.run_epoch(args, test,  0, args.nti, testMU, testLOGVAR, testPI, d_type='test_test', fout=fout)

        fout.write('writing to ' + ex_file + '\n')


    else:
        if test is not None:
            test_acc=run_epoch(model,args,test, 0, d_type='test', fout=fout)

    model.results=[np.array(VAL_ACC).transpose().reshape(-1,2)]+[test_acc[0]]
    save_net_int(model, args.model_out, args, predir)

    return(model)