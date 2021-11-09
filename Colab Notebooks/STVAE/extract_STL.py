from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
import os



class DL(DataLoader):
    def __init__(self, input, batch_size, num_class, num, shape, shuffle=False):
        super(DL, self).__init__(input,batch_size,shuffle)
        self.num=num
        self.num_class=num_class
        self.shape=shape


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

transform = transforms.Compose([
        transforms.ToTensor(),

    ])

train = datasets.STL10(get_pre()+'LSDA_data/STL', split='unlabeled', transform=transform, download=True)
