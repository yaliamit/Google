import argparse
from aux import process_args
from main import main_loc
import torch

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

def train_net(par_file,predir):
  net=main_loc(par_file)
  save_net(net,par_file,predir)

def run_net(par_file):
  net=main_loc(par_file)


def seq(par_file, predir, tlay=None, toldn=None):
    from mprep import get_network
    import aux
    import argparse
    import mprep
    datadirs = predir + 'Colab Notebooks/STVAE/'

    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                     description='')
    parser = aux.process_args(parser)

    f = open(datadirs + '_pars/' + par_file + '.txt', 'r')
    args = parser.parse_args(f.read().split())
    f.close()

    lnti, layers_dict = mprep.get_network(args.layers)

    fin = open(predir + 'Colab Notebooks/STVAE/_pars/' + par_file + '.txt', 'r')
    lines = [line.rstrip('\n') for line in fin]
    fin.close()

    oldn = toldn
    for i, d in enumerate(layers_dict):

        nn = d['name']

        if tlay is None or nn == tlay:
            tlay = None
            if 'final' in nn or 'input' in nn or 'drop' in nn or (i < len(layers_dict) - 1 and
                                                                  'pool' in layers_dict[i + 1]['name']):
                pass
            else:

                fout = open('t_par.txt', 'w')
                for l in lines:
                    if 'dense_final' in l and not 'hid' in l:
                        if args.embedd:
                            fout.write(l + ';parent:[' + nn + ']\n')
                            # fout.write('name:drop_f;drop:.5;parent:['+nn+']\n'+l+'\n')
                        else:
                            fout.write('name:drop_f;drop:.5;parent:[' + nn + ']\n' + l + '\n')
                    else:
                        fout.write(l + '\n')
                if (args.embedd):
                    fout.write('--embedd\n' + '--embedd_layer=' + nn + '\n')
                fout.write('--update_layers\n')
                if ('pool' in nn):
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
                train_net('t_par',predir)
                oldn = outn

    print("helo")