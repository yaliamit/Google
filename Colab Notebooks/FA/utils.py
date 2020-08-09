from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import torch
from torch import nn
from Conv_data import get_data
from pprint import pprint
import matplotlib.pyplot as plt
import time

class hinge_loss(nn.Module):
    def __init__(self, mu=1., num_class=10):
        super(hinge_loss, self).__init__()
        self.fac = mu/num_class
        self.ee = torch.eye(num_class)

    def forward(self, input, target):

        targarr = self.ee[target] > 0
        loss = torch.sum(torch.relu(1 - input[targarr])) + self.fac * torch.sum(
            torch.relu(1 + input[torch.logical_not(targarr)]))
        loss /= input.shape[0]
        return loss


class brelu(nn.Module):

  def __init__(self, min=0., max=1.):
        super(brelu, self).__init__()
        self.min=min
        self.max=max

  def forward(self,input):

    out=torch.clamp(input,self.min,self.max)

    return out

def get_cifar10_man(datapath, num_train):

    trainset = dset.CIFAR10(root=datapath, train=True, download=True)
    train_dat = (trainset.data.transpose(0,3,1,2)/255-0.5)/0.5
    train_tar = np.array(trainset.targets)

    testset = dset.CIFAR10(root=datapath, train=False, download=True)
    test_dat = (testset.data.transpose(0,3,1,2)/255-0.5)/0.5
    test_tar = np.array(testset.targets)
    
    return train_dat[:num_train], train_tar[:num_train], train_dat[num_train:], train_tar[num_train:], test_dat, test_tar

def get_data_rega():
    PARS = {}
    PARS['data_set'] = 'cifar10'
    PARS['nval'] = 5000
    PARS['num_train'] = 50000
    data = get_data(PARS)
    data = data[0] + data[1] + data[2]
    data = [data[0].transpose(0, 3, 1, 2), np.argmax(data[1], axis=1), data[2].transpose(0, 3, 1, 2),
            np.argmax(data[3], axis=1), data[4].transpose(0, 3, 1, 2), np.argmax(data[5], axis=1)]

    return data


def show_results(pars, savepath, sho=False):
  EXP_NAME='FA_{}_layerwise_{}_OPT_{}_CR_{}'.format(str(pars.fa),str(pars.layerwise),
                                                    pars.OPT,pars.CR)
  fig=plt.figure()
  lw_loss=np.load(savepath+'loss_'+EXP_NAME+'.npy')
  lw_acc=np.load(savepath+'val.acc_'+EXP_NAME+'.npy')
  lw_test_acc=np.load(savepath+'te.acc_'+EXP_NAME+'.npy')
  print(lw_test_acc)

  if pars.layerwise:
    for i in range(pars.NUM_LAYERS):
      plt.plot(lw_acc[(i*pars.epochs):((i+1)*pars.epochs)], label = 'Layer'+str(i))
  else:
    i=0
    plt.plot(lw_acc[(i*pars.epochs):((i+1)*pars.epochs)],label='Layer'+str(pars.NUM_LAYERS-1))
  lw_min=np.min(lw_acc)
  print(lw_min)
  plt.legend()
  plt.text(10,lw_min,'test_acc:'+str(lw_test_acc))
  plt.title(EXP_NAME)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.savefig(savepath+'img/acc_'+EXP_NAME+'.jpg')
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
  plt.savefig(savepath+'img/loss_'+EXP_NAME+'.jpg')
  if sho:
    fig.show()


def run_net(data, pars, savepath):
    pprint(vars(pars))

    EXP_NAME = 'FA_{}_layerwise_{}_OPT_{}_CR_{}'.format(str(pars.fa), str(pars.layerwise),
                                                        pars.OPT, pars.CR)

    fix = nn.Sequential()

    lw_loss = []
    lw_acc = []
    lw_test_acc = []

    for i in range(pars.START_LAYER, pars.NUM_LAYERS):
        lri = i if pars.layerwise else 0
        net, fix, layer = pars.get_net(lri, pars, fix, pars.layer_pars)

        if pars.layerwise or i == pars.NUM_LAYERS - 1:
            print('LAYER:%d' % i)
            print(fix)
            print(net)
            print(pars.optimizer)
            train_model(data, fix, net, pars, ep_loss=lw_loss, ep_acc=lw_acc)
            test_acc = check_accuracy(data[4], data[5], fix, net, pars)
            print('Layer: %d, te.acc = %.4f' % (i, test_acc))
            lw_test_acc.append(test_acc)
        fix.add_module('layer%d' % i, layer)
        print()

    np.save(savepath + 'loss_' + EXP_NAME, lw_loss)
    np.save(savepath + 'val.acc_' + EXP_NAME, lw_acc)
    np.save(savepath + 'te.acc_' + EXP_NAME, lw_test_acc)



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

            model.train()  # put model to training mode
            x = torch.from_numpy(train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = torch.from_numpy(train_tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
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

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            pars.optimizer.step()

            #if t % print_every == 0:
        acc = check_accuracy(val_dat, val_tar, fix, model, pars)
        print('Epoch %d, loss = %.4f, val.acc = %.4f, time=%.2f' % (e, loss.item(), acc, t1-time.time()))

        ep_loss.append(loss.item())
        ep_acc.append(acc)
        print()


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
        #print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc