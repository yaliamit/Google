import argparse
from aux import process_args
from main import main_loc
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def copy_to_content(fname,predir):
  datadirs = predir + 'Colab Notebooks/STVAE/'
  from_file=fname+'.txt'
  f=open(datadirs+'_pars/'+from_file,'r')
  g=open(datadirs+fname+'.txt','w')
  for l in f:
    g.write(l)
  g.close()
  f.close()

def copy_from_content(fname,predir):
  datadirs = predir + 'Colab Notebooks/STVAE/'
  to_file='_pars/'+fname+'.txt'
  g=open(datadirs+fname+'.txt','r')
  f=open(datadirs+to_file,'w')
  for l in g:
    f.write(l)
  f.close()
  g.close()

def show_results(pars, datadirs, LW,sho=False):


  savepath=datadirs+'save/'
  if not os.path.isdir(savepath):
      os.mkdir(savepath)
  if pars.layerwise:
      lay='True'
  elif pars.layerwise_randomize is not None:
      lay='Randomize'
  else:
      lay='False'
  EXP_NAME='FA_{}_layerwise_{}_hinge_{}'.format(str(pars.fa),lay,str(pars.hinge))
  print(EXP_NAME)
  fig=plt.figure()
  if pars.layerwise_randomize is None:
    num_layers=len(LW)
  else:
    num_layers=len(pars.layerwise_randomize)//2
   #np.load(savepath+'te.acc_'+EXP_NAME+'.npy')
  lw_test_acc=np.zeros(num_layers)
  if pars.layerwise:
    for i in range(num_layers):
      plt.plot(LW[i][0][:,0], label = 'Layer'+str(i))
      lw_test_acc[i]=LW[i][1]
  elif pars.layerwise_randomize is not None:
      num_iters=len(LW[0][0])//num_layers
      for i in range(num_layers):
          plt.plot(LW[0][0][i*num_iters:(i+1)*num_iters,0],label = 'Layer'+str(i))
          lw_test_acc[i] = LW[0][1][i]
  else:
    i=0
    plt.plot(LW[0][0][:,0],label='Layer'+str(num_layers-1))
    lw_test_acc[0]=LW[0][1]

  lw_min=np.min(LW[0][0][:,0])

  plt.legend()
  plt.text(10,lw_min,'test_acc:'+str(lw_test_acc))
  plt.title(EXP_NAME)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.savefig(savepath+'acc_'+EXP_NAME+'.jpg')
  if sho:
    fig.show()
  fig=plt.figure()
  if pars.layerwise:
      for i in range(num_layers):
        plt.plot(LW[i][0][:,1], label = 'Layer'+str(i))
  elif pars.layerwise_randomize is not None:
      num_iters=len(LW[0][0])//num_layers
      for i in range(num_layers):
          plt.plot(LW[0][0][i*num_iters:(i+1)*num_iters,1],label = 'Layer'+str(i))
  else:
    i=0
    plt.plot(LW[0][0][:,1],label='Layer'+str(num_layers-1))
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig(savepath+'loss_'+EXP_NAME+'.jpg')
  if sho:
    fig.show()


def save_net(net,par_file,predir):

  parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
        description='Variational Autoencoder with Spatial Transformation')

  parser=process_args(parser)
  datadirs = predir + 'Colab Notebooks/STVAE/'
  f = open(datadirs+par_file + '.txt', 'r')
  bb = f.read().split()
  aa = [ll for ll in bb if '#' not in ll]
  args=parser.parse_args(aa)
  f.close()
  model=net
  print(args.model_out)
  if args.model_out is not None:
      ss=args.model_out+'.pt'
  else:
      ss='network.pt'
  model.to('cpu')
  if 'Users/amit' in predir:
      torch.save({'args': args,
                  'model.state.dict': model.state_dict()}, predir + 'Colab Notebooks/STVAE/_output/' + ss)
  else:
    torch.save({'args': args,
        'model.state.dict': model.state_dict()}, predir+'Colab Notebooks/STVAE/_output/'+ss,_use_new_zipfile_serialization=False)

def train_net(par_file,predir, RESULTS, device):
  net=main_loc(par_file,device)
  RESULTS+=[net.results]
  save_net(net,par_file,predir)

def run_net(par_file, device):
  net=main_loc(par_file, device)
  return net


def seq(par_file, predir, device, tlay=None, toldn=None):
    from prep import get_network
    import aux
    import argparse
    import prep
    datadirs = predir + 'Colab Notebooks/STVAE/'

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='')
    parser = aux.process_args(parser)

    f = open(datadirs+par_file + '.txt', 'r')
    bb = f.read().split()
    aa = [ll for ll in bb if '#' not in ll]
    args = parser.parse_args(aa)

    f.close()

    lnti, layers_dict = prep.get_network(args.layers)

    fin = open(datadirs+par_file + '.txt', 'r')
    lines = [line.rstrip('\n') for line in fin if line[0] !='#']
    fin.close()
    RESULTS = []
    #break_name='pool'
    #break_name_layer='name:'+break_name+'f;pool_size:2'
    #skip_name='drop'

    skip_name1 = 'pool'
    skip_name2 = 'non_linearity'
    break_name='drop'
    break_name_layer = 'name:dropf;drop:.5'
    if not args.layerwise:
        fout = open(datadirs+'t_par.txt', 'w')
        for l in lines:
            fout.write(l + '\n')
        fout.close()
        train_net('t_par',predir,RESULTS,device)
    else:
        oldn = toldn

        for i, d in enumerate(layers_dict):

            nn = d['name']
            done=False
            if tlay is None or nn == tlay:
                tlay = None
                #if 'final' in nn or 'input' in nn or 'drop' in nn or (i < len(layers_dict) - 1 and
                #                                                      'pool' in layers_dict[i + 1]['name']):
                if 'final' in nn or 'input' in nn or break_name in nn  or (i < len(layers_dict) - 1 and
                                                                           (skip_name1 in layers_dict[i+1]['name'] or
                                                                            skip_name2 in layers_dict[i+1]['name'])):
                    pass
                else:

                    fout = open(datadirs+'t_par.txt', 'w')
                    for l in lines:
                        if 'dense_final' in l and not 'hid' in l:
                            if args.embedd:
                                fout.write(l + ';parent:[' + nn + ']\n')
                                # fout.write('name:drop_f;drop:.5;parent:['+nn+']\n'+l+'\n')
                            else:
                                    #fout.write('name:drop_f;drop:.5;parent:[' + nn + ']\n'+'name:Avg\n' + l + '\n')
                                    fout.write(break_name_layer +'\n')
                                    #fout.write('name:Avg\n')
                                    fout.write(l + '\n')

                        else:
                            if not done:
                                fout.write(l + '\n')
                                if 'name' in l and l.split(':')[1].split(';')[0]==nn:
                                    done=True
                    if (args.embedd):
                        fout.write('--embedd\n' + '--embedd_layer=' + nn + '\n')
                    fout.write('--update_layers\n')
                    if (skip_name1 in nn or skip_name2 in nn):
                        if skip_name2 in layers_dict[i - 1]['name']:
                            fout.write(layers_dict[i - 2]['name'] + '\n')
                        else:
                            fout.write(layers_dict[i - 1]['name'] + '\n')
                    else:
                        fout.write(nn + '\n')
                    fout.write('dense_final\n')

                    if oldn is not None:
                        fout.write('--cont_training\n' + '--model=' + oldn + '\n')
                    emb = 'cl'
                    if args.embedd:
                        emb = 'emb'
                    outn = 'network_' + nn + '_' + emb
                    fout.write('--model_out=' + outn + '\n' + '--out_file=OUT_' + nn + '_' + emb + '.txt\n')
                    fout.close()
                    train_net('t_par',predir, RESULTS,device)
                    oldn = outn
    #if not args.embedd:
    show_results(args,datadirs,RESULTS)
    print("helo")