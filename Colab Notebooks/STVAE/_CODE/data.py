
from __future__ import print_function

import sys
import os
import numpy as np
import h5py
import scipy.ndimage
import pylab as py
import matplotlib.colors as col
from images import deform_data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
import torch
import random




def get_stl10_unlabeled(batch_size, size=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train = datasets.STL10('./data', split='unlabeled', transform=transform, download=True)

    if size != 0 and size < len(train):
        train = Subset(train, random.sample(range(len(train)), size))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    trlen=int(size*.8)
    telen=int(size-trlen)
    trlen=trlen//batch_size
    telen=telen//batch_size
    aa=random_split(train_loader,[trlen, telen],generator = torch.Generator().manual_seed(42))
    return aa[0].dataset, None, aa[1].dataset


def get_stl10_labeled(batch_size,size=0):
    transform = transforms.Compose([
        transforms.ToTensor(),

    ])

    train = datasets.STL10('./data', split='train', transform=transform, download=True)

    test = datasets.STL10('./data', split='test', transform=transform, download=True)

    size=min(size,len(train)) if size>0 else len(train)

    train = Subset(train, random.sample(range(len(train)), size))
    test = Subset(test, random.sample(range(len(test)), len(test)))

    train_loader = DataLoader(train, batch_size=batch_size)

    test_loader = DataLoader(test, batch_size=batch_size)


    return train_loader, None, test_loader




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

def quantize(images,levels):

    if levels>1:
        return np.digitize(images, np.arange(levels) /levels) - 1
    else:
        return images

def enlarge(x_in,new_dim):

    h=x_in.shape[2]
    w=x_in.shape[3]

    if h>new_dim or w > new_dim:
        x_out=x_in
    else:
        dh=(new_dim-h)//2
        dw=(new_dim-w)//2
        nn=x_in.shape[0]
        nc=x_in.shape[1]
        x_out=np.zeros((nn,nc,new_dim,new_dim))


        aa=np.concatenate((np.random.randint(0,new_dim-h,(nn,1)),np.random.randint(0,new_dim-w,(nn,1))),axis=1)
        for i,x in enumerate(x_in):
            x_out[i,:,aa[i,0]:aa[i,0]+h,aa[i,1]:aa[i,1]+w]=x

    return x_out


def get_data_pre(args,dataset):


    PARS = {}
    PARS['data_set'] = dataset
    PARS['num_train'] = args.num_train // args.mb_size * args.mb_size
    PARS['nval'] = args.nval
    PARS['mb_size']=args.mb_size
    if args.cl is not None:
        PARS['one_class'] = args.cl

    train, val, test, image_dim = get_data(PARS)
    if type(train) is DataLoader:
        return [train,val,test]

    if (False): #args.edges):
        train=[pre_edges(train[0],dtr=args.edge_dtr).transpose(0,3,1,2),np.argmax(train[1], axis=1)]
        test=[pre_edges(test[0],dtr=args.edge_dtr).transpose(0,3,1,2),np.argmax(test[1], axis=1)]
        if val is not None:
            val = [pre_edges(val[0],dtr=args.edge_dtr).transpose(0, 3, 1, 2), np.argmax(val[1], axis=1)]
    else:
        if train[1].ndim>1:
            trl=np.argmax(train[1], axis=1)
            tel=np.argmax(test[1],axis=1)
            if val is not None:
              vall=np.argmax(val[1],axis=1)
        else:
            trl=train[1]
            tel=test[1]
            if val is not None:
                vall=val[1]
        train = [enlarge(quantize(train[0].transpose(0, 3, 1, 2),args.image_levels),args.new_dim),trl]
        test = [enlarge(quantize(test[0].transpose(0, 3, 1, 2), args.image_levels),args.new_dim), tel]
        if val is not None:
            val = [enlarge(quantize(val[0].transpose(0, 3, 1, 2),args.image_levels),args.new_dim), vall]

    if args.edges:
        ed = Edge(device, dtr=.03).to(device)
        edges=[]
        jump=10000
        for j in np.arange(0,train[0].shape[0],jump):
            tr=torch.from_numpy(train[0][j:j+jump]).to(device)
            edges+=[ed(tr).cpu().numpy()]
        train=[np.concatenate(edges,axis=0),train[1]]
        edges_te=[]
        for j in np.arange(0,test[0].shape[0],jump):
            tr=torch.from_numpy(test[0][j:j+jump]).to(device)
            edges_te+=[ed(tr).cpu().numpy()]
        test=[np.concatenate(edges_te,axis=0),test[1]]
        if val is not None:
            edges_va = []
            for j in np.arange(0, test[0].shape[0], jump):
                tr = torch.from_numpy(val[0][j:j + jump]).to(device)
                edges_va += [ed(tr).cpu().numpy()]
            val = [np.concatenate(edges_va,axis=0), val[1]]
    if (args.num_test>0):
        ntest=test[0].shape[0]
        ii=np.arange(0, ntest, 1)
        np.random.shuffle(ii)
        test=[test[0][ii[0:args.num_test]], test[1][ii[0:args.num_test]]]
    elif (args.num_test<0):
        test=[train[0][0:10000],train[1][0:10000]]
    print('In get_data_pre: num_train', train[0].shape[0])
    print('Num test:',test[0].shape[0])
    #train=DataLoader(list(zip(train[0],train[1])),batch_size=args.mb_size)
    #if val[0] is not None:
    #    val=DataLoader(list(zip(val[0],val[1])),batch_size=args.mb_size)
    #else:
    #    val=None
    #test=DataLoader(list(zip(test[0],test[1])),batch_size=args.mb_size)
    return [train, val, test]



def rotate_dataset_rand(X,angle=0,scale=0,shift=0,gr=0,flip=False,blur=False,saturation=False, spl=None):
    # angle=NETPARS['trans']['angle']
    # scale=NETPARS['trans']['scale']
    # #shear=NETPARS['trans']['shear']
    # shift=NETPARS['trans']['shift']
    s=np.shape(X)
    Xr=np.zeros(s)
    cent=np.array(s[1:3])/2
    angles=np.random.rand(Xr.shape[0])*angle-angle/2.
    #aa=np.int32(np.random.rand(Xr.shape[0])*.25)
    #aa[np.int32(len(aa)/2):]=aa[np.int32(len(aa)/2):]+.75
    #angles=aa*angle-angle/2
    SX=np.exp(np.random.rand(Xr.shape[0],2)*scale-scale/2.)
    SH=np.int32(np.round(np.random.rand(Xr.shape[0],2)*shift)-shift/2)
    FL=np.zeros(Xr.shape[0])
    BL=np.zeros(Xr.shape[0])
    HS=np.zeros(Xr.shape[0])
    if (flip):
        FL=(np.random.rand(Xr.shape[0])>.5)
    if (blur):
        BL=(np.random.rand(Xr.shape[0])>.5)
    if (saturation):
        HS=(np.power(2,np.random.rand(Xr.shape[0])*4-2))
        HU=((np.random.rand(Xr.shape[0]))-.5)*.2
    #SHR=np.random.rand(Xr.shape[0])*shear-shear/2.
    for i in range(Xr.shape[0]):
        # if (np.mod(i,1000)==0):
        #     print(i,end=" ")
        mat=np.eye(2)
        #mat[1,0]=SHR[i]
        mat[0,0]=SX[i,0]
        mat[1,1]=SX[i,0]
        rmat=np.eye(2)
        a=angles[i]*np.pi/180.
        rmat[0,0]=np.cos(a)
        rmat[0,1]=-np.sin(a)
        rmat[1,0]=np.sin(a)
        rmat[1,1]=np.cos(a)
        mat=mat.dot(rmat)
        offset=cent-mat.dot(cent)+SH[i]
        for d in range(X.shape[3]):
            Xt=scipy.ndimage.interpolation.affine_transform(X[i,:,:,d],mat, offset=offset, mode='reflect')
            Xt=np.minimum(Xt,.99)
            if (FL[i]):
                Xt=np.fliplr(Xt)
            if (BL[i]):
                Xt=scipy.ndimage.gaussian_filter(Xt,sigma=.5)
            Xr[i,:,:,d]=Xt
        if (HS[i]):
            y=col.rgb_to_hsv(Xr[i])
            y[:,:,1]=np.minimum(y[:,:,1]*HS[i],1)
            y[:,:,0]=np.mod(y[:,:,0]+HU[i],1.)
            z=col.hsv_to_rgb(y)
            Xr[i]=z

    if (gr):
        fig1=py.figure(1)
        fig2=py.figure(2)
        ii=np.arange(0,X.shape[0],1)
        np.random.shuffle(ii)
        nr=9
        nr2=nr*nr
        for j in range(nr2):
            #print(angles[ii[j]]) #,SX[i],SH[i],FL[i],BL[i])
            py.figure(fig1.number)
            py.subplot(nr,nr,j+1)
            py.imshow(X[ii[j]])
            py.axis('off')
            py.figure(fig2.number)
            py.subplot(nr,nr,j+1)
            py.imshow(Xr[ii[j]])
            py.axis('off')
        py.show()
    print(end="\n")

    return(np.float32(Xr))


def one_hot(values,PARS=None,n_values=10):

    n_v = np.maximum(n_values,np.max(values) + 1)
    oh=np.float32(np.eye(n_v)[values])
    if (PARS is not None):
        if ("L2" in PARS):
            oh=2.*oh-1.
    return oh



def load_dataset(pad=0,nval=10000, F=False):
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 28, 28, 1)

        if (pad>0):
            new_data=np.zeros((data.shape[0],data.shape[1],data.shape[2]+2*pad,data.shape[3]+2*pad))
            new_data[:,:,pad:pad+28,pad:pad+28]=data
            data=new_data
        # The inputs come as bytes, we convert them to floatX in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)

        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    pre=get_pre()+'LSDA_data/'
    # We can now download and read the training and test set images and labels.

    fold='mnist/'
    
    X_train = load_mnist_images(pre+fold+'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(pre+fold+'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(pre+fold+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(pre+fold+'t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    if (nval>0):
        X_train, X_val = X_train[:-nval], X_train[-nval:]
        y_train, y_val = y_train[:-nval], y_train[-nval:]
    else:
        X_val=None
        y_val=None
    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.
def get_mnist(PARS):
    if ('nval' in PARS):
        nval=PARS['nval']
    else:
        nval=10000
    F='f' in PARS['data_set']
    tr, trl, val, vall, test, testl = load_dataset(nval=nval,F=F)
    if ('one_class' in PARS):
        tr=tr[trl==PARS['one_class']]
        trl=trl[trl==PARS['one_class']]
        test = test[testl == PARS['one_class']]
        testl = testl[testl == PARS['one_class']]
        if (nval>0):
            val = val[vall == PARS['one_class']]
            vall = vall[vall == PARS['one_class']]
    trl=one_hot(trl)
    if (nval>0):
        vall=one_hot(vall)
    testl=one_hot(testl)
    return (tr,trl), (val,vall), (test,testl)

def get_cifar(PARS):

    data_set=PARS['data_set']
    pre=get_pre()+'LSDA_data/CIFAR/'

    filename = pre+data_set+'_train.hdf5'
    print(filename)
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    tr = f[key]
    print('tr',tr.shape)
    key = list(f.keys())[1]
    tr_lb=f[key]
    ntr=len(tr)-PARS['nval']
    train_data=np.float32(tr[0:ntr])/255.
    train_labels=one_hot(np.int32(tr_lb[0:ntr]),PARS)
    val=None
    if PARS['nval']:
        val_data=np.float32(tr[ntr:])/255.
        val_labels=one_hot(np.int32(tr_lb[ntr:]),PARS)
        val=(val_data,val_labels)
    filename = pre+data_set+'_test.hdf5'
    f = h5py.File(filename, 'r')
    key = list(f.keys())[0]
    # Get the data
    test_data = np.float32(f[key])/255.
    key = list(f.keys())[1]
    test_labels=one_hot(np.int32(f[key]),PARS)
    return (train_data, train_labels), val , (test_data, test_labels)

def get_letters(PARS):

    data_set=PARS['data_set']
    pre=get_pre()+'LSDA_data/mnist/'

    filename = data_set+'.npy'
    print(filename)
    train_data=np.load(pre+data_set+'_data.npy')
    train_data=np.float32(train_data.reshape((-1,28,28,1)))
    print(train_data.shape)
    if 'binarized' not in data_set:
        train_data=np.float32(train_data/255.)
    train_labels=np.load(pre+data_set+'_labels.npy')
    test_data=train_data[-10000:]
    train_data=train_data[:-10000]
    test_labels=train_labels[-10000:]
    train_labels=train_labels[:-10000]
    val=None
    if PARS['nval']:
        val_data=train_data[-PARS['nval']:]
        train_data=train_data[:-PARS['nval']]
        val_labels=train_labels[-PARS['nval']:]
        train_labels=train_labels[:-PARS['nval']]
        val=(val_data,val_labels)
    return (train_data, train_labels), val, (test_data, test_labels)



def get_data(PARS):
    if 'stl' in PARS['data_set']:
        if 'unlabeled' in PARS['data_set']:
            train,val,test=get_stl10_unlabeled(PARS['mb_size'],size=PARS['num_train'])
        else:
            train, val, test = get_stl10_labeled(PARS['mb_size'], size=PARS['num_train'])
        return train,val,test,train.dataset[0][0].shape[0]
    elif ('cifar' in PARS['data_set']):
        train, val, test=get_cifar(PARS)
    else:
        train, val, test = get_letters(PARS)
    num_train = np.minimum(PARS['num_train'], train[0].shape[0])
    train = (train[0][0:num_train], train[1][0:num_train])
    dim = train[0].shape[1]
    PARS['nchannels'] = train[0].shape[3]
    PARS['n_classes'] = np.max(train[1])+1
    print('n_classes', PARS['n_classes'], 'dim', dim, 'nchannels', PARS['nchannels'])
    return train, val, test, dim





