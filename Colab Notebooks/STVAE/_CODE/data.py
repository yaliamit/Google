
from __future__ import print_function

import sys
import os
import numpy as np
import h5py
import scipy.ndimage
import pylab as py
import matplotlib.colors as col
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split
import torch
import random
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.data import _utils

class _MultiProcessingDataLoaderIterWithIndices(_MultiProcessingDataLoaderIter):
    def __init__(self,loader):
        super(_MultiProcessingDataLoaderIterWithIndices,self).__init__(loader)

    def _next_data(self):

        while True:
            # If the worker responsible for `self._rcvd_idx` has already ended
            # and was unable to fulfill this task (due to exhausting an `IterableDataset`),
            # we try to advance `self._rcvd_idx` to find the next valid index.
            #
            # This part needs to run in the loop because both the `self._get_data()`
            # call and `_IterableDatasetStopIteration` check below can mark
            # extra worker(s) as dead.
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:  # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                if not self._persistent_workers:
                    self._shutdown_workers()
                raise StopIteration

            # Now `self._rcvd_idx` is the batch index we want to fetch

            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data), self._rcvd_idx

            assert not self._shutdown and self._tasks_outstanding > 0
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            #if self._dataset_kind == _DatasetKind.Iterable:
                # Check for _IterableDatasetStopIteration
            if isinstance(data, _utils.worker._IterableDatasetStopIteration):
                    if self._persistent_workers:
                        self._workers_status[data.worker_id] = False
                    else:
                        self._mark_worker_as_unavailable(data.worker_id)
                    self._try_put_index()
                    continue

            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data), idx

class _SingleProcessDataLoaderIterWithIndices(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(_SingleProcessDataLoaderIterWithIndices, self).__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0


    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        if self._pin_memory:
            data = _utils.pin_memory.pin_memory(data)
        return data, index

class DL(DataLoader):
    def __init__(self, input, batch_size, num_class, num, shape, num_workers=0, shuffle=False):
        super(DL, self).__init__(input,batch_size,shuffle,num_workers=num_workers)
        self.num=num
        self.num_class=num_class
        self.shape=shape



    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIterWithIndices(self)
        else:
             self.check_worker_number_rationality()
             return _MultiProcessingDataLoaderIterWithIndices(self)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, transform, n_views=2, double_aug=False):
        self.transform = transform
        self.n_views = n_views
        self.base_transform=transforms.Compose([transforms.ToTensor()])
        self.double_aug=double_aug

    def __call__(self, x):
        if self.n_views>1:
            if self.double_aug:
                return [self.transform(x), self.transform(x)]
            else:
                return [self.base_transform(x), self.transform(x)]
        else:
            return self.base_transform(x)


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        # """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # color_jitter = transforms.ColorJitter(0.4 * s, 0.4 * s, 0.4 * s, 0.1 * s)
        # data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
        #                                       transforms.RandomHorizontalFlip(),
        #                                       transforms.RandomApply([color_jitter], p=0.8),
        #                                       transforms.RandomGrayscale(p=0.2),
        #                                       transforms.ToTensor()])

        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=0.1 * 32, prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]
        )

        return data_transforms


def get_stl10_unlabeled(batch_size, size=0, crop=0):

    test=None
    test_loader=None
    if crop == 0:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        transform=ContrastiveLearningViewGenerator(
            ContrastiveLearningDataset.get_simclr_pipeline_transform(crop),
            2)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.RandomCrop(crop)])

    train = datasets.STL10(get_pre()+'LSDA_data/STL', split='unlabeled', transform=transform, download=True)
    num_class=len(train.classes)
    shape=train.data.shape[1:]
    if crop>0:
        shape=[train.data.shape[1],crop,crop]
    if size != 0 and size <= len(train):
        train = Subset(train, random.sample(range(len(train)), size))
    trlen = size
    #telen = int(size - trlen)
    #[train,test]=random_split(train,[trlen, telen])

    train_loader = DL(train, batch_size=batch_size, num_class=num_class, num=trlen, shape=shape, shuffle=True)
    #if test is not None:
    #    test_loader=DL(test, batch_size=batch_size,  num_class=num_class, num=telen, shape=shape, shuffle=True)
    return train_loader, None, test_loader


def get_stl10_labeled(batch_size,size=0,crop=0, jit=0):

    if crop > 0:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=jit,hue=jit,saturation=jit,contrast=jit)
        ])
        test_transform=transforms.Compose([
            transforms.ToTensor(), transforms.CenterCrop(crop)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor()])
        test_transform = transforms.Compose([
            transforms.ToTensor()])

    train = datasets.STL10(get_pre()+'LSDA_data/STL', split='train', transform=train_transform, download=True)
    num_class = len(train.classes)
    shape = train.data.shape[1:]
    if crop>0:
        shape=[train.data.shape[1],crop,crop]
    test = datasets.STL10(get_pre()+'LSDA_data/STL', split='test', transform=test_transform, download=True)

    size=min(size,len(train)) if size>0 else len(train)
    numtr=size
    numte=len(test) if size>0 and size==len(train) else size
    train = Subset(train, random.sample(range(len(train)), size))
    test = Subset(test, random.sample(range(len(test)), numte))

    train_loader = DL(train, batch_size=batch_size, num_class=num_class, num=numtr, shape=shape, shuffle=False)

    test_loader = DL(test, batch_size=batch_size, num_class=num_class, num=numte, shape=shape, shuffle=False)


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
    PARS['crop']=args.crop
    PARS['h_factor']=args.h_factor
    PARS['jit']=args.h_factor
    PARS['thr']=args.threshold
    PARS['double_aug']=args.double_aug

    if args.cl is not None:
        PARS['one_class'] = args.cl

    train, val, test, image_dim = getf_data(PARS)
    if type(train) is DL or dataset=='stl10':
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
    num_class=np.max(train[1])+1
    train=DL(list(zip(train[0],train[1])),batch_size=args.mb_size, num_class=num_class,
             num=train[0].shape[0], shape=train[0].shape[1:],shuffle=True)
    if val is not None:
        val=DL(list(zip(val[0],val[1])),batch_size=args.mb_size, num_class=num_class,
             num=val[0].shape[0], shape=val[0].shape[1:],shuffle=True)
    test = DL(list(zip(test[0], test[1])), batch_size=args.mb_size, num_class=num_class,
               num=test[0].shape[0], shape=test[0].shape[1:],shuffle=False)
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

def get_CIFAR10(batch_size = 500,size=None, double_aug=True):

    transform_CIFAR = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=0.1 * 32)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    transform=ContrastiveLearningViewGenerator(transform_CIFAR, double_aug=double_aug)
    train = datasets.CIFAR10(root = "data",train = True,download = True, transform = transform)
    test = datasets.CIFAR10(root = "data",train = False,download = True, transform = transform)

    num_class = len(train.classes)

    shape = list(np.array(train.data.shape[1:])[[2,0,1]])

    if size is not None and size <= len(train):
        train = Subset(train, random.sample(range(len(train)), size))
    else:
        size=len(train)
    CIFAR10_train_loader = DL(train,batch_size,num_class,size,shape)
    CIFAR10_test_loader = DL(test,batch_size,num_class,len(test),shape)

    return CIFAR10_train_loader,CIFAR10_test_loader

def get_CIFAR100(batch_size = 500, size=None, double_aug=True):


    transform_CIFAR = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=0.1 * 32)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    transform = ContrastiveLearningViewGenerator(transform_CIFAR, double_aug=double_aug)
    train = datasets.CIFAR100(root = "data",train = True,download = True, transform = transform)
    test = datasets.CIFAR100(root = "data",train = False,download = True, transform = transform)
    num_class = len(train.classes)
    shape = list(np.array(train.data.shape[1:])[[2,0,1]])
    if size is not None and size <= len(train):
        train = Subset(train, random.sample(range(len(train)), size))
    else:
        size = len(train)

    aa = os.uname()
    numworkers=0
    if 'bernie' in aa[1]:
        numworkers=12
    CIFAR100_train_loader = DL(train,batch_size,num_class,size,shape,num_workers=numworkers)
    CIFAR100_test_loader = DL(test,batch_size,num_class,len(test),shape,num_workers=numworkers)

    return CIFAR100_train_loader,CIFAR100_test_loader


def get_cifar(PARS):

    data_set=PARS['data_set']
    pre=get_pre()+'LSDA_data/CIFAR/'
    ftr=data_set.split('_')[0]
    filename = pre+ftr+'_train.hdf5'
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
    if 'def' in data_set:
        file_name=pre+data_set+'_data.npy'
        test_data = np.load(file_name)
        file_name = pre + data_set + '_labels.npy'
        test_labels=np.load(file_name)
        test_labels=one_hot(test_labels)
    else:
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
        if PARS['thr'] is not None:
            train_data=np.float32(train_data>PARS['thr'])

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


    if ('one_class' in PARS):
        train_data=train_data[train_labels==PARS['one_class']]
        train_labels=train_labels[train_labels==PARS['one_class']]
        test_data = test_data[test_labels == PARS['one_class']]
        test_labels = test_labels[test_labels == PARS['one_class']]
        if (PARS['nval']>0):
            val_data = val_data[val_labels == PARS['one_class']]
            val_labels = val_labels[val_labels == PARS['one_class']]
    if PARS['nval']>0:
        val = (val_data, val_labels)
    return (train_data, train_labels), val, (test_data, test_labels)

def get_cifar_trans(PARS):
    val=None
    ftr = PARS['data_set'].split('_')[1]
    if ftr=='trans10':
        tr,te=get_CIFAR10(PARS['mb_size'],double_aug=PARS['double_aug'])
    else:
        tr,te=get_CIFAR100(PARS['mb_size'],double_aug=PARS['double_aug'])

    return tr,val,te





def get_data(PARS):
    if 'stl' in PARS['data_set']:
        if 'unlabeled' in PARS['data_set']:
            train,val,test=get_stl10_unlabeled(PARS['mb_size'],size=PARS['num_train'],crop=PARS['crop'])

        else:
            train, val, test = get_stl10_labeled(PARS['mb_size'], size=PARS['num_train'],crop=PARS['crop'],  jit=PARS['jit'])
        return train, val, test, train.shape[0]
    elif 'cifar_trans' in PARS['data_set']:
            train,val,test=get_cifar_trans(PARS)
            return train, val, test, train.shape[0]
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





