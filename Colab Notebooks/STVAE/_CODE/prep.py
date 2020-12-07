import torch

import numpy as np
import sys
import os
from mix import STVAE_mix
from mix_by_class import STVAE_mix_by_class
import pprint
from get_net_text import get_network
import network
import argparse
import aux
from data import get_pre



def get_names(args):

    if 'Linux' in os.uname():
        predir = '/ME/My Drive/'
    else:
        predir = '/Users/amit/Google Drive/'

    datadirs = predir + 'Colab Notebooks/STVAE/'
    fout=args.fout
    ARGS = []
    STRINGS = []
    EX_FILES = []
    SMS = []
    if (args.run_existing):
        # This overides model file name
        names = args.model
        for i, name in enumerate(names):
            sm = torch.load(datadirs+'_output/' + name + '.pt',map_location='cpu')
            SMS += [sm]
            if ('args' in sm):
                args = sm['args']
            ARGS += [args]
            strings, ex_file = args.model_out, args.out_file #process_strings(args)
            STRINGS += [strings]
            EX_FILES += [ex_file]
    else:
        ARGS.append(args)
        strings, ex_file = args.model_out, args.out_file #process_strings(args)
        STRINGS += [strings]
        EX_FILES += [ex_file]
    # Print old and new arglists and compare
    if (ARGS[0] == args):
        fout.write('Printing Args from read in model and input args\n')
        pprint.pprint(vars(ARGS[0]), fout)
        # fout.write(str(ARGS[0]) + '\n')
    else:
        fout.write('Printing Args from read-in model\n')
        dcA = vars(ARGS[0])
        dca = vars(args)
        pprint.pprint(dcA, fout)
        different_items = {k: [v, dca[k]] for k, v in dcA.items() if k in dca and v != dca[k]}
        print('Difference in model args and input args', file=fout)
        pprint.pprint(different_items, fout)

    return ARGS, STRINGS, EX_FILES, SMS


def get_models(device, fout, sh,STRINGS,ARGS, args):


    models = []
    if 'vae' in args.type:
        for strings, args in zip(STRINGS, ARGS):
            model=make_model(args, sh[1:], device, fout)
            models += [model]
    elif args.network:
        # parse the existing network coded in ARGS[0]
        arg = ARGS[0]
        if args.cont_training:  # Parse the new network
            arg = args
        # Layers defining the new network.
        if arg.layers is not None:
            arg.lnti, arg.layers_dict = get_network(arg.layers, nf=sh[1])
            # Initialize the network
            models = [network.network(device, arg, arg.layers_dict, arg.lnti, fout=fout, sh=sh).to(device)]

    return models


def  make_model(args, sh, device, fout):
        model = STVAE_mix(sh, device, args).to(device)
        tot_pars = 0
        for keys, vals in model.state_dict().items():
            fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
            tot_pars += np.prod(np.array(vals.shape))
        fout.write('tot_pars,' + str(tot_pars) + '\n')
        return model

def setups(par_file):


    predir = get_pre()
    datadirs = predir + 'Colab Notebooks/STVAE/'

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='VAEs, classification and embedding networks')

    parser = aux.process_args(parser)
    f = open(par_file + '.txt', 'r')
    args = parser.parse_args(f.read().split())
    f.close()
    args.datadirs = datadirs

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_gpu = args.gpu and torch.cuda.is_available()
    if not args.CONS:
        print(args.out_file)
        fout = open(args.datadirs+args.out_file, 'a+')
    else:
        fout = sys.stdout

    args.fout=fout
    return args


def copy_from_old_to_new(model, args, fout, SMS, strings,device, sh):
    if 'vae' in args.type:
        print('device')
        model_old=make_model(args, sh[1:], device, fout)
    else:
        lnti, layers_dict = get_network(SMS['args'].layers, nf=sh[1])
        model_old = network.network(device, SMS['args'], layers_dict, lnti, fout=fout, sh=sh, first=2).to(device)
    model_old.load_state_dict(SMS['model.state.dict'])

    params_old = model_old.named_parameters()
    params = model.named_parameters()
    dict_params = dict(params)
    # Loop over parameters of N1
    print('cont training:', args.cont_training)
    for name, param_old in params_old:
        if name in dict_params and (args.update_layers is None or name.split('.')[1] not in args.update_layers):
            fout.write('copying ' + name + '\n')
            dict_params[name].data.copy_(param_old.data)
            if model_old.bn=='full':
              if 'norm' in name and 'weight' in name:
                tname=name.split('.')[1]
                sname=".".join(name.split('.')[0:2])
                msname=sname+'.running_mean'
                dict_params[msname]=getattr(model_old.layers,tname).running_mean.data
                msname =sname + '.running_var'
                dict_params[msname]=getattr(model_old.layers,tname).running_var.data
                msname = sname + '.num_batches_tracked'
                dict_params[msname]=getattr(model_old.layers, tname).num_batches_tracked.data
    if 'crit.pos_weight' in SMS['model.state.dict']:
        dict_params['crit.pos_weight']=SMS['model.state.dict']['crit.pos_weight']
    model.load_state_dict(dict_params)
    return model


