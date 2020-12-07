from make import train_model, test_models
from class_on_hidden import pre_train_new
import prep
import numpy as np
from data import get_data_pre
from images import make_images, make_sample


def main_loc(par_file, device):
  model_out = None

  args = prep.setups(par_file)
  fout=args.fout
  # reinit means you are taking part of an existing network as fixed and updating some other parts.
  if args.cont_training:
      args.run_existing=True


  ARGS, STRINGS, EX_FILES, SMS = prep.get_names(args)

  # Get data
  DATA=get_data_pre(args,args.dataset)
  args.num_class=np.int(np.max(DATA[0][1])+1)
  ARGS[0].num_class=args.num_class
  print('NUMCLASS',args.num_class)


  # Training an autoencoder.
  sh=DATA[0][0].shape

  models=prep.get_models(device, fout, sh, STRINGS, ARGS, args)
  # Training a feedforward embedding or classification network

  lr=args.lr
  fout.flush()

  if args.cont_training:

      prep.copy_from_old_to_new(models[0], args, fout, SMS[0], STRINGS[0], device, sh)
      train_model(models[0], args, EX_FILES[0], DATA, fout)
      model_out=models[0]
      pre_train_new(model_out,args,device,fout, data=None)

  elif args.run_existing:
      models[0].load_state_dict(SMS[0]['model.state.dict'])
      models[0].to(device)
      if args.sample:
          LLG=models[0].compute_likelihood(DATA[2][0], 250)
          print('LLG:',LLG,file=fout)
          make_sample(models[0],args, EX_FILES[0], datadirs=args.datadirs)
          make_images(DATA[2],models[0],EX_FILES[0],ARGS[0],datadirs=args.datadirs)
      elif args.network:
          if args.embedd:
              models[0].embedd_layer = args.embedd_layer
          pre_train_new(models[0], args, device, fout, data=DATA)
      else: # Test a sequence of models
              test_models(ARGS,SMS,DATA[2],models, fout)
      model_out=models[0]

  else: # Totally new network
        train_model(models[0], args, EX_FILES[0], DATA, fout)
        pre_train_new(models[0], args, device, fout, data=DATA)

        model_out=models[0]


  print('DONE')
  fout.flush()
  if (not args.CONS):
          fout.close()
  return model_out



