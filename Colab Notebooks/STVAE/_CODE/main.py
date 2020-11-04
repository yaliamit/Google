from models_make import train_model, test_models
import os
from class_on_hidden import pre_train_new
import network
import mprep
import numpy as np
import argparse
import aux
from get_net_text import get_network
import pprint
from Conv_data import get_pre
import torch

def copy_from_old_to_new(model, args, fout, SMS, device, sh):

    lnti, layers_dict = mprep.get_network(SMS['args'].layers, nf=sh[1])
    model_old = network.network(device, SMS['args'], layers_dict, lnti, fout=fout, sh=sh, first=2).to(device)
    model_old.load_state_dict(SMS['model.state.dict'])

    params_old = model_old.named_parameters()
    params = model.named_parameters()
    dict_params = dict(params)
    # Loop over parameters of N1
    print('In reinit, cont training:', args.cont_training)
    for name, param_old in params_old:
        if name in dict_params and (args.cont_training or name.split('.')[1] not in args.update_layers):
            fout.write('copying ' + name + '\n')
            dict_params[name].data.copy_(param_old.data)
    if 'crit.pos_weight' in SMS['model.state.dict']:
        dict_params['crit.pos_weight']=SMS['model.state.dict']['crit.pos_weight']
    model.load_state_dict(dict_params)
    return model



def main_loc(par_file, device):

  predir=get_pre()


  datadirs = predir + 'Colab Notebooks/STVAE/'
  model_out=None
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
      description='Variational Autoencoder with Spatial Transformation')

  parser=aux.process_args(parser)
  f=open(par_file+'.txt','r')
  args=parser.parse_args(f.read().split())
  f.close()
  args.datadirs=datadirs

  # reinit means you are taking part of an existing network as fixed and updating some other parts.
  if args.rerun or args.reinit:
      args.run_existing=True

  ARGS, STRINGS, EX_FILES, SMS = mprep.get_names(args)
  # Get data device and output file
  fout= mprep.setups(args, EX_FILES)
  args.fout=fout
  # Get data
  DATA=mprep.get_data_pre(args,args.dataset)
  args.num_class=np.int(np.max(DATA[0][1])+1)
  ARGS[0].num_class=args.num_class
  print('NUMCLASS',args.num_class)
  # Training an autoencoder.
  sh=DATA[0][0].shape
  if 'vae' in args.type:
      print('device')
      models=mprep.get_models(device, fout, sh,STRINGS,ARGS,locals())
  # Training a feedforward embedding or classification network
  if args.network:
      # parse the existing network coded in ARGS[0]
      arg=ARGS[0]
      if args.reinit: # Parse the new network
          arg=args
      # Layers defining the new network.
      if arg.layers is not None:
          arg.lnti, arg.layers_dict = get_network(arg.layers,nf=sh[1])
          # Initialize the network
          models = [network.network(device, arg, arg.layers_dict, arg.lnti, fout=fout, sh=sh).to(device)]


  lr=args.lr
  network_flag=args.network
  
  if (ARGS[0]==args):
      fout.write('Printing Args from read in model and input args\n')
      pprint.pprint(vars(ARGS[0]),fout)
      #fout.write(str(ARGS[0]) + '\n')
  else:
      fout.write('Printing Args from read-in model\n')
      fout.write(str(ARGS[0]) + '\n')
      fout.write('Printing Args from input args\n')
      fout.write(str(args) + '\n')

  fout.flush()

  if args.reinit:
      copy_from_old_to_new(models[0], args, fout, SMS[0], device, sh)
      train_model(models[0], args, EX_FILES[0], DATA, fout)
      model_out=models[0]
      if args.embedd:
              pre_train_new(model_out,args,device,fout, data=None)

  elif args.run_existing:

      if args.sample:
          models[0].load_state_dict(SMS[0]['model.state.dict'])
          aux.make_images(DATA[2],models[0],EX_FILES[0],args)
      elif network_flag:
          if 'vae' in args.type:
              models[0].load_state_dict(SMS[0]['model.state.dict'])
              if args.hid_layers is not None:
                  pre_train_new(models[0], args, device, fout, data=DATA)
          elif args.embedd:
              models[0].load_state_dict(SMS[0]['model.state.dict'])
              models[0].embedd_layer = args.embedd_layer
              pre_train_new(models[0],args,device, fout)
          else: # Test a sequence of models
              test_models(ARGS,SMS,DATA[2],models, fout)
      model_out=models[0]
  else: # Totally new network
      if 'vae' in args.type:
          train_model(models[0], args, EX_FILES[0], DATA, fout)
          pre_train_new(models[0], args, device, fout, data=DATA)
      else:
          train_model(models[0],args,EX_FILES[0],DATA,fout)
          if args.embedd:
                  res=pre_train_new(models[0],args,device, fout)
      model_out=models[0]


  print('DONE')
  fout.flush()
  if (not args.CONS):
          fout.close()
  return model_out



