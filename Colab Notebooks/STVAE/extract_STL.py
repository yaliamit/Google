from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
import os
import numpy as np
import random

class DL(DataLoader):
    def __init__(self, input, batch_size, num_class, num, shape, shuffle=False):
        super(DL, self).__init__(input,batch_size,shuffle)
        self.num=num
        self.num_class=num_class
        self.shape=shape


def get_pre():
    aa=os.uname()
    if 'Linux' in aa:
        if 'bernie' in aa[1] or 'aoc' in aa[1]:
            pre='/home/amit/ga/Google/'
        elif 'midway' in aa[1]:
            pre='/home/yaliamit/Google/'
        else:
            pre = 'ME/MyDrive/'
    else:
        pre = '/Users/amit/Google Drive/'

    return pre

def extract_sub_images(numtr,pr):
    train = datasets.STL10(get_pre() + 'LSDA_data/STL', split='unlabeled', transform=transform, download=True)

    batch_size=1000
    shape = train.data.shape[1:]
    train = Subset(train, range(numtr))
    DATA = DL(train, batch_size=batch_size, num_class=0, num=numtr, shape=shape, shuffle=True)

    II=[]
    size=32
    for bb in enumerate(DATA):
        ii=np.random.randint(size/2,size/2+size,[DATA.batch_size,pr,2])
        for k,b in enumerate(bb[1][0]):
            for j in range(pr):
                II+=[np.expand_dims(b[:,ii[k][j,0]:ii[k][j,0]+size,ii[k][j,1]:ii[k][j,1]+size].numpy(),axis=0)]

    print(len(II))
    III=np.concatenate(II)

    np.save('stl_unlabeled_sub',III)

transform = transforms.Compose([
        transforms.ToTensor(),

    ])





extract_sub_images(1000,3)
