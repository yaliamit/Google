
from __future__ import print_function

import sys
import os
import platform
import numpy as np
import h5py
import scipy.ndimage
import pylab as py
import matplotlib.colors as col
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, random_split, Dataset
import torch
from images import Edge, create_img
from imageio import imsave
import random
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from torch.utils.data import _utils, SubsetRandomSampler
import PIL
import pylab as py

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
        super(DL, self).__init__(input,batch_size,shuffle=shuffle,num_workers=num_workers)
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
            return self.transform(x)


import numpy as np
from PIL import ImageFilter


class GaussianBlur(object):
    """Implementation of random Gaussian blur.
    Utilizes the built-in ImageFilter method from PIL to apply a Gaussian
    blur to the input image with a certain probability. The blur is further
    randomized as the kernel size is chosen randomly around a mean specified
    by the user.
    Attributes:
        kernel_size:
            Mean kernel size for the Gaussian blur.
        prob:
            Probability with which the blur is applied.
        scale:
            Fraction of the kernel size which is used for upper and lower
            limits of the randomized kernel size.
    """

    def __init__(self, kernel_size: float, prob: float = 0.5,
                 scale: float = 0.2):
        self.prob = prob
        self.scale = scale
        # limits for random kernel sizes
        self.min_size = (1 - scale) * kernel_size
        self.max_size = (1 + scale) * kernel_size
        self.kernel_size = kernel_size

    def __call__(self, sample):
        """Blurs the image with a given probability.
        Args:
            sample:
                PIL image to which blur will be applied.

        Returns:
            Blurred image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            # choose randomized kernel size
            kernel_size = np.random.normal(
                self.kernel_size, self.scale * self.kernel_size
            )
            kernel_size = max(self.min_size, kernel_size)
            kernel_size = min(self.max_size, kernel_size)
            radius = int(kernel_size / 2)
            # return blurred image
            return sample.filter(ImageFilter.GaussianBlur(radius=radius))
        # return original image
        return sample

def get_simclr_pipeline_transform(size=32,factor=1.):

        if factor>0:
            data_transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.8*factor, 0.8*factor, 0.8*factor, 0.2*factor)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                #GaussianBlur(kernel_size=.1*size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
            )
        else:
            data_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
            ])
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
            get_simclr_pipeline_transform(crop),
            2)
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.RandomCrop(crop)])

    train = datasets.STL10(os.path.join(get_pre(),'LSDA_data','STL'), split='unlabeled', transform=transform, download=True)
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

    train = datasets.STL10(os.path.join(get_pre(),'LSDA_data','STL'), split='train', transform=train_transform, download=True)
    num_class = len(train.classes)
    shape = train.data.shape[1:]
    if crop>0:
        shape=[train.data.shape[1],crop,crop]
    test = datasets.STL10(os.path.join(get_pre(),'LSDA_data','STL'), split='test', transform=test_transform, download=True)

    size=min(size,len(train)) if size>0 else len(train)
    numtr=size
    numte=len(test) if size>0 and size==len(train) else size
    train = Subset(train, random.sample(range(len(train)), size))
    test = Subset(test, random.sample(range(len(test)), numte))

    train_loader = DL(train, batch_size=batch_size, num_class=num_class, num=numtr, shape=shape, shuffle=False)

    test_loader = DL(test, batch_size=batch_size, num_class=num_class, num=numte, shape=shape, shuffle=False)


    return train_loader, None, test_loader

def get_pre():
    aa=platform.system()
    bb=platform.node()
    if 'Linux' in aa:
        if 'bernie' in bb or 'aoc' in bb:
            pre=os.path.join('/home','amit','ga','Google')
        else:
            pre = os.path.join('ME','MyDrive')
    elif 'Windows' in aa:
        pre=os.path.join('C:\\','Users','amit','My Drive')
    else:
        pre = os.path.join('/Users','amit','Google Drive')
        if os.path.isdir(os.path.join(pre,'My Drive')):
            pre=os.path.join(pre,'My Drive')

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

    pre=os.path.join(get_pre(),'LSDA_data')
    # We can now download and read the training and test set images and labels.

    fold='mnist'
    
    X_train = load_mnist_images(os.path.join(pre,fold,'train-images-idx3-ubyte.gz'))
    y_train = load_mnist_labels(os.path.join(pre,fold,'train-labels-idx1-ubyte.gz'))
    X_test = load_mnist_images(os.path.join(pre,fold,'t10k-images-idx3-ubyte.gz'))
    y_test = load_mnist_labels(os.path.join(pre,fold,'t10k-labels-idx1-ubyte.gz'))

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

def cifar10_train_classifier_transforms(input_size=32):
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

class SimpleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, input, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = input[0]
        self.labels=input[1]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample =  PIL.Image.fromarray(np.uint8(255*self.data[idx]),mode="RGB")
        lab = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return (sample, lab)
def get_CIFAR10(PARS, batch_size = 500,size=None, double_aug=True, factor=1., emb=True, val_num=0):

    val_loader=None
    PARS['data_set']='cifar'+PARS['data_set'].split('_')[1][5:]
    tr, vl, ts =get_cifar(PARS)
    transform_CIFAR = get_simclr_pipeline_transform(factor=factor)

    numworkers = 0
    aa = os.uname()
    if 'bernie' in aa[1] or 'aoc' in aa[1]:
        numworkers = 12
    if emb:
        transform=ContrastiveLearningViewGenerator(transform_CIFAR, double_aug=double_aug)
    else:
        transform = ContrastiveLearningViewGenerator(transform_CIFAR, n_views=1)

    train = SimpleDataset(tr, transform = transform)
    #train = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    #test = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    if val_num>0:
        val = datasets.CIFAR10(root="data", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )]))

    test = SimpleDataset(ts, transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]
             )]))

    #num_class = len(train.classes)#len(np.unique(tr[1]))
    num_class = len(np.unique(tr[1]))
    shape = list(np.array(train.data.shape[1:])[[2,0,1]])

    if size is not None and size <= len(train):
        train = Subset(train, random.sample(range(len(train)), size))
    if val_num>0:
        num_train = len(train)
        ii=np.array(range(num_train))
        np.random.shuffle(ii)
        val = Subset(val, ii[num_train-val_num:num_train])
        train=Subset(train, ii[0:num_train-val_num])

    if 'one_class' in PARS:
        train=Subset(train, [i for i, (x, y) in enumerate(train) if y == PARS['one_class']])
        if val_num>0:
            train=Subset(train, [i for i, (x, y) in enumerate(train) if y == PARS['one_class']])
    if val_num>0:
        val_loader = DL(val, batch_size, num_class, len(val), shape,num_workers = numworkers)

    CIFAR10_train_loader = DL(train,batch_size,num_class,len(train),shape,num_workers=numworkers,shuffle=True)
    CIFAR10_test_loader = DL(test,batch_size,num_class,len(test),shape,num_workers=numworkers)

    return CIFAR10_train_loader,val_loader, CIFAR10_test_loader





def get_CIFAR100(PARS, batch_size = 500, size=None, double_aug=True, factor=1., emb=True):
    transform_CIFAR = get_simclr_pipeline_transform(factor=factor)
    # PARS['data_set']="cifar100"
    # train, val, test = get_cifar(PARS)
    # tel = np.argmax(test[1], axis=1)
    # test=list(zip(test[0].transpose(0, 3, 1, 2),tel))
    # shape = list(np.array(train[0].shape[1:])[[2, 0, 1]])
    # num_class = len(np.unique(train[1]))

    if emb:
        transform = ContrastiveLearningViewGenerator(transform_CIFAR, double_aug=double_aug)
    else:
        transform = ContrastiveLearningViewGenerator(transform_CIFAR, n_views=1)
    train = datasets.CIFAR100(root="data", train=True, download=True, transform=transform)
    test = datasets.CIFAR100(root="data", train=False, download=True, transform=transform)
    shape = list(np.array(train.data.shape[1:])[[2,0,1]])
    # if size is not None and size <= len(train[0]):
    #     ii=random.sample(range(len(train[0])),size)
    #     trl = np.argmax(train[1], axis=1)
    #     train=list(zip(train[0][ii].transpose(0, 3, 1, 2),trl[ii]))
    #     #train = Subset(zip(train), random.sample(range(len(train[0])), size))
    # else:
    #      size = len(train[0])
    #      trl = np.argmax(train[1], axis=1)
    #      train=list(zip(train[0],trl[1]))

    aa = os.uname()
    numworkers=0
    if 'bernie' in aa[1] or 'aoc' in aa[1]:
        numworkers=12
    CIFAR100_train_loader = DL(train,batch_size,num_class,size,shape,num_workers=numworkers,shuffle=True)
    CIFAR100_test_loader = DL(test,batch_size,num_class,len(test),shape,num_workers=numworkers)

    return CIFAR100_train_loader,CIFAR100_test_loader


def get_cifar(PARS):

    data_set=PARS['data_set']
    pre=os.path.join(get_pre(),'LSDA_data','CIFAR')
    ftr=data_set.split('_')[0]
    filename = os.path.join(pre,ftr+'_train.hdf5')
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
    tr = train_data[0:100]
    tr = tr.transpose(0, 3, 1, 2)
    img = create_img(tr, tr[0].shape)
    py.imshow(img)
    py.show()
    py.imshow(tr[10].transpose(1,2,0))
    py.show()
    train_labels=tr_lb[0:ntr] #one_hot(np.int32(tr_lb[0:ntr]),PARS)
    val=None
    if PARS['nval']:
        val_data=np.float32(tr[ntr:])/255.
        val_labels=tr_lb[ntr:] #one_hot(np.int32(tr_lb[ntr:]),PARS)
        val=(val_data,val_labels)
    if 'def' in data_set:
        file_name=os.path.join(pre,data_set,'_data.npy')
        test_data = np.load(file_name)
        file_name = os.path.join(pre,data_set + '_labels.npy')
        test_labels=np.load(file_name)
        #test_labels=one_hot(test_labels)
    else:
        filename = os.path.join(pre,data_set+'_test.hdf5')
        f = h5py.File(filename, 'r')
        key = list(f.keys())[0]
        # Get the data
        test_data = np.float32(f[key])/255.
        key = list(f.keys())[1]
        test_labels=f[key] #one_hot(np.int32(f[key]),PARS)
    return (train_data, train_labels), val , (test_data, test_labels)

def get_letters(PARS):

    data_set=PARS['data_set']
    pre=os.path.join(get_pre(),'LSDA_data','mnist')

    filename = data_set+'.npy'
    print(filename)
    train_data=np.load(os.path.join(pre,data_set+'_data.npy'))
    train_data=np.float32(train_data.reshape((-1,28,28,1)))
    print(train_data.shape)
    if 'binarized' not in data_set:
        train_data=np.float32(train_data/255.)
        if PARS['thr'] is not None:
            train_data=np.float32(train_data>PARS['thr'])

    train_labels=np.load(os.path.join(pre,data_set+'_labels.npy'))
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



    if PARS['nval']>0:
        val = (val_data, val_labels)
    return (train_data, train_labels), val, (test_data, test_labels)

def get_cifar_trans(PARS):
    val=None
    ftr = PARS['data_set'].split('_')[1]
    if ftr=='trans10':
        tr,val,te=get_CIFAR10(PARS, PARS['mb_size'],size=PARS['num_train'],double_aug=PARS['double_aug'],factor=PARS['h_factor'],emb=PARS['emb'], val_num=PARS['nval'])
    else:
        tr,te=get_CIFAR100(PARS, PARS['mb_size'],size=PARS['num_train'],double_aug=PARS['double_aug'],factor=PARS['h_factor'],emb=PARS['emb'])

    return tr,val,te

from scipy.ndimage import gaussian_filter

def get_synth(PARS):


    dat=np.random.randn(PARS['num_train']+PARS['nval']+PARS['num_train'],32,32).astype(np.float32)

    for i in range(len(dat)):
        dat[i]=gaussian_filter(dat[i],sigma=8,mode='wrap')
    dat=dat.reshape(-1,32,32,1)
    #dat=dat/np.max(dat)
    train=(dat[0:PARS['num_train']],np.zeros(PARS['num_train']))
    if PARS['nval']>0:
        val=(dat[PARS['num_train']:PARS['num_train']+PARS['nval']],np.zeros(PARS['nval']))
    else:
        val=None
    test=(dat[PARS['num_train']+PARS['nval']:PARS['num_train']+PARS['num_train']+PARS['nval']],np.zeros(PARS['num_train']))
    print("hello")
    return train,val,test


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
    elif 'synth' in PARS['data_set']:
        train,val,test=get_synth(PARS)
    else:
        train, val, test = get_letters(PARS)
    num_train = np.minimum(PARS['num_train'], train[0].shape[0])
    train = (train[0][0:num_train], train[1][0:num_train])
    if ('one_class' in PARS):
        tr=train[0][train[1]==PARS['one_class']]
        trl=train[1][train[1]==PARS['one_class']]
        train=(tr,trl)
        te = test[0][test[1] == PARS['one_class']]
        tel= test[1][test[1] == PARS['one_class']]
        test=(te,tel)
        if (PARS['nval']>0):
            va= val[0][val[1] == PARS['one_class']]
            val = val[1][val[1] == PARS['one_class']]
            val=(va,val)
    dim = train[0].shape[1]
    PARS['nchannels'] = train[0].shape[3]
    PARS['n_classes'] = np.max(train[1])+1
    print('n_classes', PARS['n_classes'], 'dim', dim, 'nchannels', PARS['nchannels'])
    return train, val, test, dim




def get_data_pre(args,dataset,device):


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
    PARS['emb']= True if args.embedd_type is not None else False
    if args.cl is not None:
        PARS['one_class'] = args.cl

    train, val, test, image_dim = get_data(PARS)
    if type(train) is DL or dataset=='stl10':
        return [train,val,test]



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
