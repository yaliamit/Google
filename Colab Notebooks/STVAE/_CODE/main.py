from make import train_model, test_models
from class_on_hidden import pre_train_new, cluster_hidden
import prep
from data import get_data_pre
from images import make_images, show_examples_of_deformed_images, get_embs, deform_data
import pylab as py
import numpy as np
import sys

def main_loc(par_file, device,net=None):

  embed_data=None
  args = prep.setups(par_file)
  fout=args.fout
  if net is None:
    args.verbose = True


  ARGS, STRINGS, EX_FILES, SMS = prep.get_names(args)

  # Get data
  DATA=get_data_pre(args,args.dataset)

  #if type(DATA[0]) is DL:
  sh=DATA[0].shape
  args.num_class=DATA[0].num_class
  #else:
  #    sh=DATA[0][0].shape[1:]
  #    args.num_class=np.int(np.max(DATA[0][1])+1)
  ARGS[0].num_class=args.num_class
  ARGS[0].patch_size=args.patch_size
  print('NUMCLASS',args.num_class)

  # Training an autoencoder.



  #if net is None:
  #sh=[3,96,96]
  models=prep.get_models(device, fout, sh, STRINGS, ARGS, args)
  if args.deform:
      OUT=[]
      LL=[]
      done=False
      for bb in enumerate(DATA[2]):
          if not done:
            show_examples_of_deformed_images(bb[1],args)
            done=True
          out = deform_data(bb[1][0], args.perturb, args.transformation, args.s_factor, args.h_factor, False)
          OUT+=[out.numpy()]
          LL+=[bb[1][1].numpy()]
      LLA=np.concatenate(LL)
      OUTA = np.concatenate(OUT, axis=0).transpose(0,2,3,1)
      np.save('cifar10_def_data',OUTA)
      np.save('cifar10_def_labels',LLA)
      sys.exit()
  #   return models[0], embed_data, args
  # else:
  #   models=[net]
  fout.flush()
  model_out=models[0]

  if args.cont_training and not args.run_existing:
      prep.copy_from_old_to_new(models[0], args, fout, SMS[0], STRINGS[0], device, sh)
      models[0].nti = args.nti
      models[0].opt_jump = args.opt_jump
      train_model(models[0], args, EX_FILES[0], DATA, fout)
      model_out = models[0]
      if args.hid_nepoch>0:
        pre_train_new(model_out,args,device,fout, data=None)

  elif args.run_existing:
      models[0].load_state_dict(SMS[0]['model.state.dict'])
      models[0].to(device)

      if args.sample:
          oldopt=models[0].opt
          models[0].opt=args.OPT
          models[0].mu_lr=args.mu_lr

          if args.show_weights is not None:
                ww=getattr(models[0].enc_conv.model.back_layers,args.show_weights).weight.data
                for w in ww[0]:
                    py.imshow(w,cmap='gray')
                    py.show()
          models[0].nti=args.nti
          #LLG=models[0].compute_likelihood(DATA[2][0], 250)
          #print('LLG:',LLG,file=fout)
          #make_sample(models[0],args, EX_FILES[0], datadirs=args.datadirs)
          make_images(DATA[2],models[0],EX_FILES[0],args,datadirs=args.datadirs)
          models[0].opt=oldopt
      elif args.network and (args.embedd or 'ae' in args.type):
          if args.embedd:
              models[0].embedd_layer = args.embedd_layer
          pre_train_new(models[0], args, device, fout, data=DATA)
      elif args.cluster_hidden:
          cluster_hidden(models[0], args, device, DATA, args.datadirs, EX_FILES[0])
      else: # Test a sequence of models
          ARGS[0].nti = args.nti
          test_models(ARGS,SMS,DATA[2],models, fout)
  else: # Totally new network

        train_model(models[0], args, EX_FILES[0], DATA, fout)
        if args.hid_nepoch>0:
            embed_data=pre_train_new(models[0], args, device, fout, data=DATA)
        model_out=models[0]


  print('DONE')
  fout.flush()
  if (not args.CONS):
          fout.close()
  return model_out, embed_data, args



