import torch

import numpy as np
import sys
import os
from data import get_data
from images import Edge
from get_net_text import get_network

def process_strings(args):
    strings={'opt_pre':'', 'mm_pre':'', 'opt_post':'', 'opt_mix':'', 'opt_class':'', 'cll':''}
    if (args.OPT):
        strings['opt_pre']='OPT_'
        strings['opt_post']='_OPT'
    if (args.only_pi):
        strings['opt_pre'] = 'PI_'
        strings['opt_post'] = '_PI'
    if (args.n_mix>=1):
        strings['opt_mix']='_mix'
    if (args.n_class>0):
        strings['opt_class']='_by_class'

    if (args.cl is not None):
        strings['cll']=str(args.cl)
    ex_file = args.out_file.split('.')[0]
    #OUT_fistrings['opt_pre'] + strings['opt_class'] + args.type + '_' + args.transformation + \
    #           '_mx_' + str(args.n_mix) + '_sd_' + \
    #          str(args.sdim) + '_cl_' + strings['cll']
    return strings, ex_file




def get_names(args):

    if 'Linux' in os.uname():
        predir = '/ME/My Drive/'
    else:
        predir = '/Users/amit/Google Drive/'

    datadirs = predir + 'Colab Notebooks/STVAE/'

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
            strings, ex_file = process_strings(args)
            STRINGS += [strings]
            EX_FILES += [ex_file]
    else:
        ARGS.append(args)
        strings, ex_file = process_strings(args)
        STRINGS += [strings]
        EX_FILES += [ex_file]

    return ARGS, STRINGS, EX_FILES, SMS





def get_models(device, fout, sh,STRINGS,ARGS, locs):

    from mix import STVAE_mix
    from mix_by_class import STVAE_mix_by_class

    models = []
    for strings, args in zip(STRINGS, ARGS):
        model=make_model(strings,args,locals(), sh[1:], device, fout)
        models += [model]

    return models



def  make_model(strings,args,locs, sh, device, fout):
        model = locs['STVAE' + strings['opt_mix'] + strings['opt_class']](sh, device, args).to(device)
        tot_pars = 0
        for keys, vals in model.state_dict().items():
            fout.write(keys + ',' + str(np.array(vals.shape)) + '\n')
            tot_pars += np.prod(np.array(vals.shape))
        fout.write('tot_pars,' + str(tot_pars) + '\n')
        return model

def setups(args, EX_FILES):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    use_gpu = args.gpu and torch.cuda.is_available()
    if not args.CONS:
        print(args.out_file)
        fout = open(args.datadirs+args.out_file, 'a+')
    else:
        fout = sys.stdout


    return fout



