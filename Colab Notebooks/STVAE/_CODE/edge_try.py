
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from data import get_data

class Edge(torch.nn.Module):
    def __init__(self, device, ntr=4, dtr=0):
        super(Edge, self).__init__()
        self.ntr = ntr
        self.dtr = dtr
        self.dv = device
        self.marg=2
        self.delta=3
        self.dirs=[(1,1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0),(-1,1),(1,-1)]
        self.slope=10


    def gt(self,x):

        #y=torch.gt(x,0)
        y=torch.sigmoid(x*self.slope)
        return y

    def forward(self, x):
        x = self.pre_edges(x).to(self.dv)
        return x


    def pre_edges(self, im):

        ED=self.get_edges(im)
            # Loop through the 3 channels separately.

        return ED

    def get_edges(self,im):

        sh=im.shape
        delta=self.delta

        im_a=torch.cat([torch.zeros(sh[0],sh[1],sh[2],delta),im,torch.zeros(sh[0],sh[1],sh[2],delta)],dim=3)
        im_b=torch.cat([torch.zeros(sh[0],sh[1],delta,im_a.shape[3]),im_a,torch.zeros(sh[0],sh[1],delta,im_a.shape[3])],dim=2)





        diff_11 = torch.roll(im_b,(1,1),dims=(2,3))-im_b
        diff_nn11 = torch.roll(im_b, (-1, -1) ,dims=(2,3)) - im_b

        diff_01 = torch.roll(im_b,(0,1), dims=(2,3))-im_b
        diff_n01 = torch.roll(im_b,(0,-1),dims=(2,3))-im_b
        diff_10 = torch.roll(im_b,(1,0), dims=(2,3))-im_b
        diff_n10 = torch.roll(im_b,(-1,0),dims=(2,3))-im_b
        diff_n11 = torch.roll(im_b,(-1,1),dims=(2,3))-im_b
        diff_1n1 = torch.roll(im_b,(1,-1),dims=(2,3))-im_b


        thresh=self.ntr
        dtr=self.dtr
        ad_10=torch.abs(diff_10)
        ad_10=ad_10*self.gt(ad_10-dtr).float()
        e10a=self.gt(ad_10-torch.abs(diff_01)).type(torch.float)\
              + self.gt(ad_10-torch.abs(diff_n01)).type(torch.float) + self.gt(ad_10-torch.abs(diff_n10)).type(torch.float)
        e10b=self.gt(ad_10-torch.abs(torch.roll(diff_01,(1,0),dims=(1,2)))).type(torch.float)+\
                     self.gt(ad_10-torch.abs(torch.roll(diff_n01, (1, 0), dims=(1, 2)))).type(torch.float)+\
                             self.gt(ad_10-torch.abs(torch.roll(diff_01, (1, 0), dims=(1, 2)))).type(torch.float)
        e10 = self.gt(e10a+e10b-thresh) * self.gt(diff_10)
        e10n =self.gt(e10a+e10b-thresh) * self.gt(-diff_10)

        ad_01 = torch.abs(diff_01)
        ad_01 = ad_01*self.gt(ad_10-dtr).float()
        e01a = self.gt(ad_01-torch.abs(diff_10)).type(torch.float) \
               + self.gt(ad_01-torch.abs(diff_n10)).type(torch.float) + self.gt(ad_01-torch.abs(diff_n01)).type(torch.float)
        e01b = self.gt(ad_01-torch.abs(torch.roll(diff_10, (0, 1), dims=(1, 2)))).type(torch.float) + \
                self.gt(ad_01-torch.abs(torch.roll(diff_n10, (0, 1), dims=(1, 2)))).type(torch.float) +\
                    self.gt(ad_01-torch.abs(torch.roll(diff_01, (0, 1), dims=(1, 2)))).type(torch.float)
        e01 = self.gt(e01a + e01b-thresh) * self.gt(diff_01)
        e01n = self.gt(e01a + e01b-thresh) * self.gt(diff_01)



        ad_11 = torch.abs(diff_11)
        ad_11 = ad_11*self.gt(ad_11-dtr).float()
        e11a = self.gt(ad_11-torch.abs(diff_n11)).type(torch.float) \
               + self.gt(ad_11-torch.abs(diff_1n1)).type(torch.float) + self.gt(ad_11-torch.abs(diff_nn11)).type(torch.float)
        e11b = self.gt(ad_11-torch.abs(torch.roll(diff_n11, (1, 1), dims=(1, 2)))).type(torch.float) + \
                self.gt(ad_11-torch.abs(torch.roll(diff_1n1, (1, 1), dims=(1, 2)))).type(torch.float) + \
                    self.gt(ad_11-torch.abs(torch.roll(diff_11, (1, 1), dims=(1, 2)))).type(torch.float)
        e11 = self.gt(e11a + e11b-thresh) * self.gt(diff_11)
        e11n = self.gt(e11a + e11b-thresh) * self.gt(-diff_11)


        ad_n11 = torch.abs(diff_n11)
        ad_n11 = ad_n11 * (ad_n11 > dtr).float()

        en11a= self.gt(ad_n11-torch.abs(diff_11)).type(torch.float) \
               + self.gt(ad_n11-torch.abs(diff_1n1)).type(torch.float) + self.gt(ad_n11-torch.abs(diff_nn11)).type(torch.float)
        en11b = self.gt(ad_n11-torch.abs(torch.roll(diff_11, (-1, 1), dims=(1, 2)))).type(torch.float) + \
               self.gt(ad_n11-torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.float) + \
               self.gt(ad_n11-torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.float)
        en11 = self.gt(en11a + en11b-thresh) * self.gt(diff_n11)
        en11=en11.type(torch.float)
        en11n = self.gt(en11a + en11b-thresh) * self.gt(-diff_n11)
        en11n=en11n.type(torch.float)

        marg=self.marg

        edges=torch.zeros(sh[0],sh[1],8,sh[2],sh[3]).to(self.dv)
        edges[:,:,0,marg:sh[2]-marg,marg:sh[3]-marg]=e10[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,1,marg:sh[2]-marg,marg:sh[3]-marg]=e10n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,2,marg:sh[2]-marg,marg:sh[3]-marg]=e01[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,3,marg:sh[2]-marg,marg:sh[3]-marg]=e01n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,4,marg:sh[2]-marg,marg:sh[3]-marg]=e11[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,5,marg:sh[2]-marg,marg:sh[3]-marg]=e11n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,6,marg:sh[2]-marg,marg:sh[3]-marg]=en11[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,7,marg:sh[2]-marg,marg:sh[3]-marg]=en11n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]


        edges=edges.reshape(-1,8*sh[1],sh[2],sh[3])
        return(edges)


PARS={}
PARS['data_set'] = "cifar10"
PARS['num_train'] = 1000
PARS['nval'] = 0
PARS['mb_size']=100
PARS['thr']=None

train, val, test, image_dim = get_data(PARS)

dd=torch.from_numpy(train[0].transpose(0,3,1,2))
dd.requires_grad=True
sh=dd.shape
ee=Edge(torch.device('cpu'))

eaa=ee(dd[0:2])

s=torch.sum(eaa)

s.backward()

print('hello')
