
   
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange, tqdm

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_pre():
    aa=os.uname()
    if 'Linux' in aa:
        if 'bernie' in aa[1]:
            pre='/home/amit/ga/Google/'
        elif 'midway' in aa[1]:
            pre='/home/yaliamit/Google/'
        else:
            pre = 'ME/MyDrive/'
    else:
        pre = '/Users/amit/Google Drive/'

    return pre


predir=get_pre()
datadirs=predir+'Colab Notebooks/STVAE/'
sys.path.insert(1, datadirs)
sys.path.insert(1, datadirs+'_CODE')

print(sys.argv)
if not torch.cuda.is_available():
    device=torch.device("cpu")
else:
    if len(sys.argv)==1:
        s="cuda:"+"0"
    else:
        s="cuda:"+sys.argv[1]
    device=torch.device(s)
print(device)

datadir=predir+'LSDA_data/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_mnist():

    
    data=np.float64(np.load(datadir+'mnist/MNIST_data.npy'))
    labels=np.float32(np.load(datadir+'mnist/MNIST_labels.npy'))
    print(data.shape)
    data=np.float32(data)/255.
    train_dat=data[0:55000].reshape((-1,1,28,28))
    train_labels=np.int32(labels[0:55000])
    val_dat=data[55000:60000].reshape((-1,1,28,28))
    val_labels=np.int32(labels[55000:60000])
    test_dat=data[60000:70000].reshape((-1,1,28,28))
    test_labels=np.int32(labels[60000:70000])
    
    return (train_dat, train_labels), (val_dat, val_labels), (test_dat, test_labels)


# ### Get cifar10 data and split into training, validation and testing.

# In[3]:


def get_cifar():
    tr=np.float32(np.load(datadir+'CIFAR/cifar10_train.npy')).transpose(0,3,1,2)
    tr_lb=np.int32(np.load(datadir+'CIFAR/cifar10_train_labels.npy'))
    train_data=tr[0:45000]/255.
    train_labels=tr_lb[0:45000]
    val_data=tr[45000:]/255.
    val_labels=tr_lb[45000:]
    test_data=np.float32(np.load(datadir+'CIFAR/cifar10_test.npy')).transpose(0,3,1,2)
    test_data=test_data/255.
    test_labels=np.int32(np.load(datadir+'CIFAR/cifar10_test_labels.npy'))
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


# ### Get the data 

# In[4]:


def get_data(data_set):
    if (data_set=="mnist"):
        return(get_mnist())
    elif (data_set=="cifar"):
        return(get_cifar())


# ### The network

# In[ ]:

criterion=nn.CrossEntropyLoss()

def get_optimizer(model,pars):
    if pars.minimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=pars.step_size)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=pars.step_size)

    return optimizer




class MNIST_Net(nn.Module):
    def __init__(self,pars):
        super(MNIST_Net, self).__init__()
        

        self.mid_layer=pars.mid_layer
        # Two successive convolutional layers.
        # Two pooling layers that come after convolutional layers.
        # Two dropout layers.
        self.conv1 = nn.Conv2d(1, 32, kernel_size=pars.kernel_size[0],padding=pars.kernel_size[0]//2)
        self.pool1=nn.MaxPool2d(kernel_size=[pars.pool_size],stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=pars.kernel_size[1],padding=pars.kernel_size[1]//2)
        self.drop2 = nn.Dropout2d(pars.dropout)
        self.pool2=nn.MaxPool2d(kernel_size=2,stride=2)
        self.drop_final=nn.Dropout(pars.dropout)
       

        
    def forward(self, x, pars):
        
        # Apply relu to a pooled conv1 layer.
        x = F.relu(self.pool1(self.conv1(x)))
        if pars.first:
            print('conv1',x.shape)
        # Apply relu to a pooled conv2 layer with a drop layer inbetween.
        x = self.drop2(F.relu(self.pool2(self.conv2(x))))
        if pars.first:
            print('conv2',x.shape)
        
        if pars.first:
            pars.first=False
            pars.inp=x.shape[1]*x.shape[2]*x.shape[3]
            # Compute dimension of output of x and setup a fully connected layer with that input dim 
            # pars.mid_layer output dim. Then setup final 10 node output layer.
            print('input dimension to fc1',pars.inp)
            if pars.mid_layer is not None:
                self.fc1 = nn.Linear(pars.inp, pars.mid_layer)
                self.fc_final = nn.Linear(pars.mid_layer, 10)
            else:
                self.fc1=nn.Identity()
                self.fc_final = nn.Linear(pars.inp, 10)
            # Print out all network parameter shapes and compute total:
            tot_pars=0
            for k,p in self.named_parameters():
                tot_pars+=p.numel()
                print(k,p.shape)
            print('tot_pars',tot_pars)
        x = x.reshape(-1, pars.inp)
        x = F.relu(self.fc1(x))
        x = self.drop_final(x)
        x = self.fc_final(x)
        return x
    
    # Run the network on the data, compute the loss, compute the predictions and compute classification rate/

    
    # Compute classification and loss and then do a gradient step on the loss.

def get_acc_and_loss(model, data, targ, pars):
    output = model.forward(data,pars)
    loss = criterion(output, targ)
    pred = torch.max(output, 1)[1]
    correct = torch.eq(pred, targ).sum()

    return loss, correct

def run_grad(data,targ, model, pars):
    
        loss, correct=get_acc_and_loss(model,data,targ,pars)
        pars.optimizer.zero_grad()
        loss.backward()
        pars.optimizer.step()
        
        return loss, correct
    
        


# # Run one epoch

# In[ ]:


def run_epoch(net,epoch,train,pars,num=None,ttype="train"):
    
    
    if ttype=='train':
        t1=time.time()
        n=train[0].shape[0]
        if (num is not None):
            n=np.minimum(n,num)
        ii=np.array(np.arange(0,n,1))
        np.random.shuffle(ii)
        tr=train[0][ii]
        y=train[1][ii]
        train_loss=0; train_correct=0

        for j in trange(0,n,pars.batch_size):
                
                # Transfer the batch from cpu to gpu 
                data=torch.torch.from_numpy(tr[j:j+pars.batch_size]).to(pars.device)
                targ=torch.torch.from_numpy(y[j:j+pars.batch_size]).type(torch.long).to(pars.device)
                
                # Implement SGD step on batch
                loss, correct = run_grad(data,targ,net,pars)
                
                train_loss += loss.item()
                train_correct += correct.item()
                

        train_loss /= len(y)
        print('\nTraining set epoch {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch,
            train_loss, train_correct, len(y),
            100. * train_correct / len(y)))


def net_test(net,val,pars,ttype='val'):
    net.eval()
    with torch.no_grad():
                test_loss = 0
                test_correct = 0
                vald=val[0]
                yval=val[1]
                for j in np.arange(0,len(yval),pars.batch_size):
                    data=torch.from_numpy(vald[j:j+pars.batch_size]).to(device)
                    targ = torch.from_numpy(yval[j:j+pars.batch_size]).type(torch.long).to(pars.device)
                    loss,correct=get_acc_and_loss(net,data,targ, pars)

                    test_loss += loss.item()
                    test_correct += correct.item()

                test_loss /= len(yval)
                SSS='Validation'
                if (ttype=='test'):
                    SSS='Test'
                print('\n{} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(SSS,
                    test_loss, test_correct, len(yval),
                    100. * test_correct / len(yval)))


# # Run the training. Save the model and test at the end

# In[ ]:


import time
class par(object):
    def __init__(self):
        self.batch_size=1000
        self.step_size=.001
        self.num_epochs=20
        self.numtrain=5000
        self.minimizer="Adam"
        self.data_set="mnist"
        self.model_name="model"
        self.dropout=0.
        self.dim=32
        self.pool_size=2
        self.kernel_size=5
        self.mid_layer=256
        self.use_gpu=False
        self.first=True
        self.mid_layer=None
pars=par()



pars.device = device
pars.kernel_size=[5,5]
train,val,test=get_data(data_set=pars.data_set)
pars.inp_dim=train[0][0].shape
net = MNIST_Net(pars).to(pars.device)
net.to(pars.device)
pars.optimizer=get_optimizer(net,pars)

train=(train[0][0:pars.numtrain],train[1][0:pars.numtrain])

for i in range(pars.num_epochs):
    run_epoch(net,i,train,pars, num=pars.numtrain, ttype="train")
    net_test(net,val,pars)


net_test(net,test,pars,ttype="test")

torch.save(net.state_dict(), datadir+"tmp/"+pars.model_name)

