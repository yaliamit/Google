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

from scipy.stats import rankdata
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'






def get_digits(img, verbose=False):

    digs=[]
    bordersy=[[5,95],[5,95],[5,95],[5,95],[8,95]]
    bordersx=[[0,45],[40,80],[75,120],[112,157],[150,195]]
    for bx,by in zip(bordersx,bordersy):
        im = img[by[0]:by[1], bx[0]:bx[1]]
        digs+=[im]

    return digs




def find_bounding_box(ima):

    imc = np.sum(ima, axis=0)
    dimc = np.diff(imc)
    iis = np.where(dimc > 0)[0][0]
    iie = np.where(dimc < 0)[0][-1]
    dimc = np.diff(np.sum(ima, axis=1))
    jjs = np.where(dimc > 0)[0][0]
    jje = np.where(dimc < 0)[0][-1]
    jjs = np.maximum(jjs - 1, 0)
    jje = np.minimum(jje + 1, ima.shape[0])
    iis = np.maximum(iis - 1, 0)
    iie = np.minimum(iie + 1, ima.shape[1])
    bim=ima[jjs:jje, iis:iie]
    return [jjs, jje, iis, iie], bim


def get_derivatives(fts, thresh=.02, res=None, ww=True):
    if ww:
        fph=np.array([[0,1,0],[0,1,0],[0,1,0]]).reshape(3,3)
    else:
        fph=np.array([[0,0,0],[0,1,0],[0,0,0]]).reshape(3,3)

    fpv=fph.transpose()
    ftsh = filters.farid_h(fts)

    ftsha = 1. * dilation((ftsh > thresh),fph)
    if res is not None:
        ftsha = resize(ftsha, res)
        #ftsha=filters.gaussian(ftsha,1)
    ftshb = 1. * dilation((ftsh < -thresh), fph)
    if res is not None:
        ftshb = resize(ftshb, res)
        #ftshb = filters.gaussian(ftshb, 1)
    ftsv = filters.farid_v(fts)

    ftsva = 1. * dilation((ftsv > thresh), fpv)
    if res is not None:
        ftsva = resize(ftsva, res)
        #ftsva = filters.gaussian(ftsva, 1)
    ftsvb = 1. * dilation((ftsv < -thresh), fpv)
    if res is not None:
        ftsvb = resize(ftsvb, res)
        #ftsvb = filters.gaussian(ftsvb, 1)
    return [ftsva, ftsvb, ftsha, ftshb]


def minimize_affine(x,z,num_iter=6):

    lam=0

    w = x[0].shape[0]
    h = x[0].shape[1]
    c = len(x)
    x = torch.from_numpy(np.array(x)).unsqueeze(0)
    z = torch.from_numpy(np.array(z)).unsqueeze(0)
    idty=torch.zeros(1,2,3,dtype=float)
    idty[0,0,0]=1; idty[0,1,1]=1
    u=torch.zeros(2,3,dtype=float)
    u=torch.autograd.Variable(u, requires_grad=True)
    opt=torch.optim.SGD([u],lr=.01)
    opt.zero_grad()

    for i in range(num_iter):
        theta=idty+u.unsqueeze(0)
        grid = F.affine_grid(theta, x.view(-1, c, w, h).size(), align_corners=True)
        xout = F.grid_sample(x.view(-1, c, w, h), grid, padding_mode='border', align_corners=False)

        loss=-torch.mean(z*torch.log(xout)+(1-z)*torch.log(1-xout))+lam*torch.sum(u*u)
        #print(i,loss.item(),torch.sum(u*u).item())
        loss.backward()
        opt.step()
        if i==20:
            opt.param_groups[0]['lr']=.0001
        #scheduler.step()
    return loss.detach().numpy(),xout.detach().numpy()



def classify_digs(digs, dftemps, verbose=False, slot=None):
    lab = []

    for i,di in enumerate(digs):
        #di = (di - np.min(di)) / (np.max(di) - np.min(di))
        di=rankdata(di.reshape(-1), 'average').reshape(di.shape)/di.size
        if verbose:
            py.imshow(di, cmap='gray')
            py.show()
        ddi = get_derivatives(di, thresh=.05, res=(height, width))
        dii = np.concatenate(ddi, axis=1)
        out=np.zeros((10,2))
        for j,dt, l in zip(range(10),dftemps, labels):
            dist, dtout = minimize_affine(dt, ddi)
            out[j,0]=dist
            out[j,1]=j
        top_two=out[out[:,0].argsort()]
        l0=int(top_two[0,1]); l1=int(top_two[1,1])
        dists0, dt0=minimize_affine(dftemps[l0],ddi,num_iter=10)
        dists1, dt1=minimize_affine(dftemps[l1],ddi,num_iter=10)
        lb= labels[l0] if dists0 < dists1 else labels[l1]
        if verbose and (slot is None or i==int(slot)):
            ddt = np.concatenate(list(dt0.squeeze()), axis=1)
            py.imshow(np.concatenate((dii, ddt), axis=0), cmap='gray')
            py.show()
            ddt = np.concatenate(list(dt1.squeeze()), axis=1)
            py.imshow(np.concatenate((dii, ddt), axis=0), cmap='gray')
            py.show()
            print('Top Two',labels[l0],dists0,labels[l1],dists1)
        lab += [lb]


    print(lab)
    return (lab)


def get_templates(templates):
    temps = []
    width = 0
    height = 0

    rrx = [[20, 55], [60, 90], [100, 130], [130, 170], [175, 205]]
    rry=[[25, 70],[75,120]]
    for xx in rrx:
        for yy in rry:
            im = templates[yy[0]:yy[1], xx[0]:xx[1],0]
            ima = (im > .5) * 1.
            bounds, bim=find_bounding_box(ima)
            height = np.maximum(height, bounds[1] - bounds[0])
            width = np.maximum(width, bounds[3] - bounds[2])
            temps += [bim]

    ftemps = []
    height+=4
    width+=4
    for tt in temps:
        newtt = np.zeros((height, width))
        aa = tt.shape
        iis = (height - aa[0]) // 2
        jjs = (width - aa[1]) // 2
        newtt[iis:iis + aa[0], jjs:jjs + aa[1]] = tt
        ftemps += [newtt]
    dftemps=[]

    for ft in ftemps:
        fts = 1. - ft
        dtemps=get_derivatives(fts,thresh=.1,ww=True)
        dtemps_sm=[]
        for dit in dtemps:
            dtemps_sm+=[np.clip(filters.gaussian(dit,sigma=1),.005,.8)]
        dftemps+=[dtemps_sm]
    return ftemps, dftemps, width, height


def extract_sub_image(img):

    tform = AffineTransform(shear=-.2, rotation=.27, scale=(.9, 1.), translation=(5, -10))
    tsform = AffineTransform(shear=0, rotation=-.03, scale=(1., 1.2), translation=(0, 0))
    scale = .07
    bounds = [0, 110, 20, 240]
    bounds1 = [0, 90, 0, 195]


    imgg = skimage.color.rgb2gray(img)
    img10 = rescale(imgg, scale)
    img10w = skimage.transform.warp(img10, tform)
    img10ws = img10w[bounds[0]:bounds[1], bounds[2]:bounds[3]]
    # py.imshow(img10ws,cmap='gray')
    # py.show()
    img10ss = skimage.transform.warp(img10ws, tsform)
    # py.imshow(img10ss, cmap='gray')
    # py.show()
    img10sss = img10ss[bounds1[0]:bounds1[1], bounds1[2]:bounds1[3]]
    # py.imshow(img10sss, cmap='gray')
    # py.show()
    #if verbose:
    #    py.imshow(img10sss, cmap='gray')
    #    py.show()
    return img10sss


if __name__ == '__main__':

    if sys.platform == 'linux':
        predir=''
    else:
        predir='../'

    lam=10.
    num_affine_iter=10
    templates=mpimage.imread(predir+'pics/images.png')
    fp=None #np.ones((3,3))
    ftemps, dftemps, width, height=get_templates(templates)

    labels=[0,5,1,6,2,7,3,8,4,9]

    slot=None
    f=None
    if len(sys.argv)==1:
        if sys.platform == 'linux':
            cmd='scp pi@192.168.0.196:gas_meter_pics/v* '+predir+'pics/.'
            print(cmd)
            os.system(cmd)
            cmd = 'ssh pi@192.168.0.196  "mv gas_meter_pics/*.jpg gas_meter_pics/_bak/." '
            print(cmd)
            os.system(cmd)
        aa=os.listdir(predir+'pics')
        verbose=False
        f=open(predir+'gas_readings.txt','a')
    elif sys.argv[1]=='test':
        aa = os.listdir(predir + 'pics')
        verbose=False
    else:
        if len(sys.argv)==3:
            slot=sys.argv[2]
        aa=[sys.argv[1]]
        verbose=True

    aa.sort()

    for i,a in enumerate(aa):
        if 'jpg' in a:
            print(a)
            img = mpimage.imread(predir + 'pics/' + a)

            img10sss=extract_sub_image(img)

            digs=get_digits(img10sss,verbose=verbose)

            lb=classify_digs(digs,dftemps,verbose=verbose, slot=slot)

            s=''
            for l in lb:
                s=s+' '+str(l)
            if f is not None:
                f.write(''.join(''.join(a.split('.')[0]).split('_')[1:])+':'+s+'\n')
            py.close()
    if f is not None:
        f.close()

    cmd = 'rm ' + predir + 'pics/v*'
    os.system(cmd)




