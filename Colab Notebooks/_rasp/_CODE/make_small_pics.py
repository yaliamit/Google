import PIL
import matplotlib.image as mpimage
import pylab as py
from skimage.transform import rescale, AffineTransform, resize
import skimage
from skimage import filters
from skimage.morphology import dilation
import numpy as np
import torch
import torch.nn.functional as F
import os
from ocr import extract_sub_image, get_digits

from scipy.stats import rankdata
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'







if sys.platform == 'linux':
    predir=''
else:
    predir='/Users/amit/Google Drive/Colab Notebooks/_rasp/'

lam=10.
num_affine_iter=10
fp=None #np.ones((3,3))

labels=[0,5,1,6,2,7,3,8,4,9]

slot=None
f=None

aa=os.listdir(predir+'pics')
aa.sort()
f=open(predir+'corrected_gas_readings.txt')
reads=f.readlines()
print(len(aa))
small_images=[]
isol=[[] for i in range(10)]
for iss in isol:
    for j in range(5):
        iss+=[[]]
for i,a,rr in zip(range(0,len(aa)),aa,reads):
    if 'jpg' in a:
        rrr=rr.strip('\n').split(':')[1].strip(' ').split(' ')
        print(a)
        img = mpimage.imread(predir + 'pics/' + a)
        img10sss=extract_sub_image(img)
        digs=get_digits(img10sss)
        for ll,dd,r in zip(range(5),digs,rrr):
            isol[int(r)][ll]+=[dd]

if f is not None:
    f.close()

small_images=np.array(small_images)
small_images=np.uint8(small_images*255)
np.save(predir+'_small_gas_pics/data',small_images)

for i,iss in zip(range(10),isol):
    for j,ar in zip(range(5),iss):
        if len(ar)>0:
            arr=np.array(ar)
            arr=np.uint8(arr*255)
            path=predir + '_isolated_gas_digits' + '/_' + str(i) + '/_loc' + str(j)
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path,arr)

print("Hello")




