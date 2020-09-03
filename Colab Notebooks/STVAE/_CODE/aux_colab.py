import argparse
from aux import process_args
from main import main_loc
import torch
import matplotlib.pyplot as plt
import numpy as np

def copy_to_content(fname,predir):
  datadirs = predir + 'Colab Notebooks/STVAE/'
  from_file=fname+'.txt'
  f=open(datadirs+'_pars/'+from_file,'r')
  g=open(fname+'.txt','w')
  for l in f:
    g.write(l)
  g.close()
  f.close()

def copy_from_content(fname,predir):
  datadirs = predir + 'Colab Notebooks/STVAE/'
  from_file='_pars/'+fname+'.txt'
  g=open(fname+'.txt','r')
  f=open(datadirs+from_file,'w')
  for l in g:
    f.write(l)
  f.close()
  g.close()

def show_results(pars, datadirs, LW,sho=False):

  savepath=datadirs+'save/'
  EXP_NAME='FA_{}_layerwise_{}_hinge_{}'.format(str(pars.fa),str(pars.layerwise),str(pars.hinge))
  print(EXP_NAME)
  fig=plt.figure()
  num_layers=len(LW)
   #np.load(savepath+'te.acc_'+EXP_NAME+'.npy')
  lw_test_acc=np.zeros(num_layers)
  if pars.layerwise:
    for i in range(num_layers):
      plt.plot(LW[i][0][:,0], label = 'Layer'+str(i))
      lw_test_acc[i]=LW[i][1]
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
  f=open(par_file+'.txt','r')
  args=parser.parse_args(f.read().split())
  f.close()
  model=net
  print(args.model_out)
  if args.model_out is not None:
      ss=args.model_out+'.pt'
  else:
      ss='network.pt'
  torch.save({'args': args,
        'model.state.dict': model.state_dict()}, predir+'Colab Notebooks/STVAE/_output/'+ss)

def train_net(par_file,predir, RESULTS, device):
  net=main_loc(par_file,device)
  RESULTS+=[net.results]
  save_net(net,par_file,predir)

def run_net(par_file, device):
  net=main_loc(par_file, device)
  return net


def seq(par_file, predir, device, tlay=None, toldn=None):
    from mprep import get_network
    import aux
    import argparse
    import mprep
    datadirs = predir + 'Colab Notebooks/STVAE/'

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='')
    parser = aux.process_args(parser)

    f = open(par_file + '.txt', 'r')
    args = parser.parse_args(f.read().split())
    f.close()

    lnti, layers_dict = mprep.get_network(args.layers)

    fin = open(par_file + '.txt', 'r')
    lines = [line.rstrip('\n') for line in fin]
    fin.close()
    RESULTS = []
    #break_name='pool'
    #break_name_layer='name:'+break_name+'f;pool_size:2'
    #skip_name='drop'

    skip_name = 'pool'
    break_name='drop'
    break_name_layer = 'name:dropf;drop:.5'
    if not args.layerwise:
        fout = open('t_par.txt', 'w')
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
                                                                      skip_name in layers_dict[i+1]['name']):
                    pass
                else:

                    fout = open('t_par.txt', 'w')
                    for l in lines:
                        if 'dense_final' in l and not 'hid' in l:
                            if args.embedd:
                                fout.write(l + ';parent:[' + nn + ']\n')
                                # fout.write('name:drop_f;drop:.5;parent:['+nn+']\n'+l+'\n')
                            else:
                                    #fout.write('name:drop_f;drop:.5;parent:[' + nn + ']\n'+'name:Avg\n' + l + '\n')
                                    fout.write(break_name_layer +'\n'+'name:Avg\n' + l + '\n')

                        else:
                            if not done:
                                fout.write(l + '\n')
                                if 'name' in l and l.split(':')[1].split(';')[0]==nn:
                                    done=True
                    if (args.embedd):
                        fout.write('--embedd\n' + '--embedd_layer=' + nn + '\n')
                    fout.write('--update_layers\n')
                    if (skip_name in nn):
                        fout.write(layers_dict[i - 1]['name'] + '\n')
                    else:
                        fout.write(nn + '\n')
                    fout.write('dense_final\n')

                    if oldn is not None:
                        fout.write('--reinit\n' + '--model=' + oldn + '\n')
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