import numpy as np
import torch
from torch import nn
from pprint import pprint
import matplotlib.pyplot as plt
import time
import os

class hinge_loss(nn.Module):
    def __init__(self, mu=1., num_class=10):
        super(hinge_loss, self).__init__()
        self.fac = mu/(num_class-1)
        self.ee = torch.eye(num_class)

    def forward(self, input, target):

        targarr = self.ee[target] > 0
        loss = torch.sum(torch.relu(1 - input[targarr])) + self.fac * torch.sum(
            torch.relu(1 + input[torch.logical_not(targarr)]))
        loss /= input.shape[0]
        return loss

def show_results(pars, savepath, LW,sho=False):
  EXP_NAME='FA_{}_layerwise_{}_OPT_{}_CR_{}'.format(str(pars.fa),str(pars.layerwise),
                                                    pars.OPT,pars.CR)
  fig=plt.figure()
  lw_loss=LW[0] #np.load(savepath+'loss_'+EXP_NAME+'.npy')
  lw_acc=LW[1] #np.load(savepath+'val.acc_'+EXP_NAME+'.npy')
  lw_test_acc=LW[2] #np.load(savepath+'te.acc_'+EXP_NAME+'.npy')

  if pars.layerwise:
    for i in range(pars.NUM_LAYERS):
      plt.plot(lw_acc[(i*pars.epochs):((i+1)*pars.epochs)], label = 'Layer'+str(i))
  else:
    i=0
    plt.plot(lw_acc[(i*pars.epochs):((i+1)*pars.epochs)],label='Layer'+str(pars.NUM_LAYERS-1))
  lw_min=np.min(lw_acc)

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
      for i in range(pars.NUM_LAYERS):
        plt.plot(lw_loss[(i*pars.epochs):((i+1)*pars.epochs)], label = 'Layer'+str(i))
  else:
    i=0
    plt.plot(lw_loss[(i*pars.epochs):((i+1)*pars.epochs)],label='Layer'+str(pars.NUM_LAYERS-1))
  plt.legend()
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig(savepath+'loss_'+EXP_NAME+'.jpg')
  if sho:
    fig.show()


def run_net(data, pars, savepath):
    pprint(vars(pars), stream=pars.fout)

    EXP_NAME = 'FA_{}_layerwise_{}_OPT_{}_CR_{}'.format(str(pars.fa), str(pars.layerwise),
                                                        pars.OPT, pars.CR)

    fix = nn.Sequential()

    lw_loss = []
    lw_acc = []
    lw_test_acc = []

    for i in range(pars.START_LAYER, pars.NUM_LAYERS):

        pars.i = i
        net, fix, layer = pars.get_net(pars, fix, pars.layer_pars)

        if pars.layerwise or i == pars.NUM_LAYERS - 1:
            pars.fout.write('LAYER:%d\n' % i)
            print(fix, file=pars.fout)
            print(net,file=pars.fout)
            print(pars.optimizer, file=pars.fout)
            train_model(data, fix, net, pars, ep_loss=lw_loss, ep_acc=lw_acc)
            test_acc = check_accuracy(data[4], data[5], fix, net, pars)
            print('Layer: %d, te.acc = %.4f' % (i, test_acc))
            lw_test_acc.append(test_acc)
        fix.add_module('layer%d' % i, layer)
        print()

    return [lw_loss, lw_acc, lw_test_acc]




def train_model(data, fix, model, pars, ep_loss, ep_acc):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    device=pars.device
    dtype = torch.float32
    train_dat=data[0]; train_tar=data[1]
    val_dat=data[2]; val_tar=data[3]
    fix = fix.to(device=device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(pars.epochs):
        t1=time.time()
        for j in np.arange(0,len(train_tar),pars.batch_size):
            #t2=time.time()

            model.train()  # put model to training mode
            x = torch.from_numpy(train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = torch.from_numpy(train_tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
            #print('dat', time.time() - t2)
            if pars.layerwise:
              with torch.no_grad():
                  x1 = fix(x)
            else:
              x1=fix(x)
            scores = model(x1)
            loss = pars.criterion(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            pars.optimizer.zero_grad()

            #t2 = time.time()
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            pars.optimizer.step()
            #print('batch',time.time()-t2)


        acc = check_accuracy(val_dat, val_tar, fix, model, pars)
        print('Epoch %d, loss = %.4f, val.acc = %.4f, time=%.2f\n' % (e, loss.item(), acc, time.time()-t1), file=pars.fout)

        ep_loss.append(loss.item())
        ep_acc.append(acc)
        pars.fout.flush()


def check_accuracy(dat, tar, fix, model, pars): # mode = 'val' or 'test'
        
    device=pars.device
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for j in np.arange(0,len(tar),pars.batch_size):
            x = torch.from_numpy(dat[j:j+pars.batch_size]).to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = torch.from_numpy(tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
            x1 = fix(x)
            scores = model(x1)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
    return acc