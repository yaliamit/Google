import torch

import numpy as np
import sys
from mix import STVAE_mix
from mix_by_class import STVAE_mix_by_class
import pprint
from get_net_text import get_network
import network
import argparse
import aux
from data import get_pre



def get_names(args):

    predir=get_pre()

    datadirs = predir + 'Colab Notebooks/STVAE/'
    fout=args.fout
    ARGS = []
    STRINGS = []
    EX_FILES = []
    SMS = []
    if (args.run_existing or args.cont_training):
        # This overides model file name
        names = args.model
        for i, name in enumerate(names):
            sm = torch.load(datadirs+'_output/' + name + '.pt',map_location='cpu')
            SMS += [sm]
            if ('args' in sm):
                arg = sm['args']
            ARGS += [arg]
            strings, ex_file = arg.model_out, arg.out_file #process_strings(args)
            STRINGS += [strings]
            EX_FILES += [ex_file]
    else:
        ARGS.append(args)
        strings, ex_file = args.model_out, args.out_file #process_strings(args)
        STRINGS += [strings]
        EX_FILES += [ex_file]
    # Print old and new arglists and compare
    if args.verbose:
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
            fout.write('Difference in model args and input args\n')
            pprint.pprint(different_items, fout)

        fout.flush()
    return ARGS, STRINGS, EX_FILES, SMS


def get_models(device, fout, sh,STRINGS,ARGS, args):


    models = []
    if 'ae' in args.type:
        for strings, args in zip(STRINGS, ARGS):
            model=make_model(args, sh, device, fout)
            models += [model]
    elif args.network:
        # parse the existing network coded in ARGS[0]
        arg = ARGS[0]
        if args.cont_training:  # Parse the new network
            arg = args
        # Layers defining the new network.
        if arg.layers is not None:
            lnti, layers_dict = get_network(arg.layers, nf=sh[0])
            # Initialize the network
            models = [network.network(device, arg, layers_dict, lnti, fout=fout, sh=sh).to(device)]

    return models


def  make_model(args, sh, device, fout):
        if args.n_class > 1:
           model=STVAE_mix_by_class(sh,device,args).to(device)
        else:
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
    f = open(datadirs+par_file + '.txt', 'r')
    bb=f.read().split()
    aa = [ll for ll in bb if '#' not in ll]

    args = parser.parse_args(aa)
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


def get_running_mean_var(model, name, dict_params):
    tname = name.split('.')[1]
    sname = ".".join(name.split('.')[0:2])
    msname = sname + '.running_mean'
    dict_params[msname] = getattr(model.layers, tname).running_mean.data
    msname = sname + '.running_var'
    dict_params[msname] = getattr(model.layers, tname).running_var.data
    msname = sname + '.num_batches_tracked'
    dict_params[msname] = getattr(model.layers, tname).num_batches_tracked.data
    return dict_params

def copy_from_old_to_new(model, args, fout, SMS, strings,device, sh):


    print('cont training:', args.cont_training)

    if 'ae' in args.type:
        print('device')
        #model=make_model(args, sh[1:], device, fout)
        model.load_state_dict(SMS['model.state.dict'])
        return
    else:
        ### TEMPORARY
        #SMS['args'] = args
        lnti, layers_dict = get_network(SMS['args'].layers, nf=sh[0])
        print('LOADING OLD MODEL')


        model_old = network.network(device, SMS['args'], layers_dict, lnti, fout=fout, sh=sh, first=2).to(device)
    model_old.load_state_dict(SMS['model.state.dict'])
    model_old.bn=args.bn
    params_old = model_old.named_parameters()
    params = model.named_parameters()
    dict_params = dict(params)
    # Loop over parameters of N1

    for name, param_old in params_old:
        #if 'clapp' in name:
        #    continue
        temp=name.split('.')
        temp[1]+='_fa'
        temp_name='.'.join(temp)
        if name in dict_params:
            pass
        elif temp_name in dict_params:
            name=temp_name
        else:
            continue
            #if name in dict_params or temp_name in dict_params:
        if (args.update_layers is None or 'copy' in args.update_layers or name.split('.')[1] not in args.update_layers):
                    fout.write('copying ' + name + '\n')
                    dict_params[name].data.copy_(param_old.data)
                    if 'norm' in name and 'weight' in name:
                     if hasattr(model_old,'bn') and model_old.bn=='full':
                        dict_params=get_running_mean_var(model_old,name,dict_params)
    temp_dict=dict_params.copy()
    for name, v in temp_dict.items():
            if 'norm' in name and 'weight' in name:
                if hasattr(model, 'bn') and model.bn == 'full':
                    dict_params=get_running_mean_var(model, name,dict_params)
    if 'crit.pos_weight' in SMS['model.state.dict']:
        dict_params['crit.pos_weight']=SMS['model.state.dict']['crit.pos_weight']
    model.load_state_dict(dict_params)
    return


