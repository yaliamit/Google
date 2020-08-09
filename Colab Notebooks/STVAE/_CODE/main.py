from models_make import train_model, test_models
import torch
from models_opt import STVAE_OPT
from models_mix import STVAE_mix
from models import STVAE
from models_mix_by_class import STVAE_mix_by_class
import os
from class_on_hidden import pre_train_new
import network
import mprep
import argparse
import aux
from get_net_text import get_network


def main_loc(par_file):

  if 'Linux' in os.uname():
        predir = '/ME/My Drive/'
  else:
        predir = '/Users/amit/Google Drive/'

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
  fout, device= mprep.setups(args, EX_FILES)
  args.fout=fout
  print(fout)
  DATA=mprep.get_data_pre(args,args.dataset)
  if not hasattr(args,'opt_jump'):
      args.opt_jump=1
      args.enc_conv=False

  if 'vae' in args.type:
      models=mprep.get_models(device, fout, DATA[0][0].shape,STRINGS,ARGS,locals())
  if args.network:
      sh=DATA[0][0].shape
      # parse the existing network coded in ARGS[0]
      arg=ARGS[0]
      if args.reinit: # Parse the new network
          arg=args
      if arg.layers is not None:
          arg.lnti, arg.layers_dict = get_network(arg.layers,nf=sh[1])
          model = network.network(device, arg, arg.layers_dict, arg.lnti).to(device)
          temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(device)
          bb = model.forward(temp)
          net_models = [model]
          if 'vae' not in args.type:
              models=net_models

  lr=args.lr
  network_flag=args.network
  
  if (ARGS[0]==args):
      fout.write('Printing Args from read in model and input args\n')
      fout.write(str(ARGS[0]) + '\n')
  else:
      fout.write('Printing Args from read-in model\n')
      fout.write(str(ARGS[0]) + '\n')
      fout.write('Printing Args from input args\n')
      fout.write(str(args) + '\n')

  fout.flush()

  if args.reinit:
      SMS[0]['args'].fout=fout
      lnti, layers_dict = mprep.get_network(SMS[0]['args'].layers, nf=sh[1])
      net_model_old = network.network(device, SMS[0]['args'], layers_dict, lnti).to(device)
      temp = torch.zeros(1, sh[1], sh[2], sh[3]).to(device)
      net_model_old.first=2
      bb = net_model_old.forward(temp)
      net_model_old.load_state_dict(SMS[0]['model.state.dict'])
      net_model=models[0]
      params = net_model_old.named_parameters()
      params2 = net_model.named_parameters()
      dict_params2 = dict(params2)
      # Loop over parameters of N1
      print('In reinit, cont training:',args.cont_training)
      for name, param in params:
          if name in dict_params2 and (args.cont_training or name.split('.')[1] not in args.update_layers):
              fout.write('copying '+name+'\n')
              dict_params2[name].data.copy_(param.data)
      net_model.load_state_dict(dict_params2)

      model_out=train_model(net_model, args, EX_FILES[0], DATA, fout)
      if (args.embedd and args.hid_layers is not None):
          if args.hid_dataset is not None:
              print('getting:'+args.hid_dataset)
              DATA = mprep.get_data_pre(args, args.hid_dataset)
              pre_train_new(net_model,DATA,args,device,fout)

  elif args.run_existing:

      if args.sample:
          model=models[0]
          model.load_state_dict(SMS[0]['model.state.dict'])
          aux.make_images(DATA[2],model,EX_FILES[0],args)
      elif network_flag:
          if 'vae' in args.type:
              model = models[0]
              model.load_state_dict(SMS[0]['model.state.dict'])
              dat, HVARS = aux.prepare_recons(model, DATA, args,fout)
              if args.hid_layers is not None:
                  pre_train_new(args, HVARS[0], HVARS[2], device, fout)
          elif args.embedd:
              args.type='net'
              args.lr=lr
              net_model=net_models[0]
              net_model.load_state_dict(SMS[0]['model.state.dict'])
              #cc=net_model.get_binary_signature(DATA[0])
              net_model.embedd_layer = args.embedd_layer
              pre_train_new(net_model,DATA,args,device, fout)
          else: # Test a sequence of models
              if args.layers is not None and not args.rerun:
                  args.type='net'
                  test_models(ARGS,SMS,DATA[2],net_models, fout)

  else: # Totally new network
      if 'vae' in args.type:
          train_model(models[0], args, EX_FILES[0], DATA, fout)
          dat,HVARS=aux.prepare_recons(models[0],DATA,args,fout)
          if args.hid_layers is not None:
                  pre_train_new(args, HVARS[0], HVARS[2], device)
      else:
          net_model=net_models[0]
          model_out=train_model(net_model,args,EX_FILES[0],DATA,fout)
          if args.model_out is not None:
            ss=args.model_out+'.pt'
          else:
            ss='network.pt'
          #torch.save({'args': args,
          #          'model.state.dict': model.state_dict()}, '/ME/My Drive/Colab Notebooks/STVAE/_output/'+ss)
          if args.embedd:
              if args.hid_layers is not None:
                  print('getting:' + args.hid_dataset)
                  args.num_train=args.network_num_train
                  DATA = mprep.get_data_pre(args, args.hid_dataset)
                  pre_train_new(net_model,DATA,args,device, fout)



  print('DONE')
  fout.flush()
  if (not args.CONS):
          fout.close()
  return model_out



