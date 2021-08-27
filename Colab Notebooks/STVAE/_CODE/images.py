import torch
import numpy as np
import os
import torch.nn.functional as F
from imageio import imsave
from scipy import ndimage

def make_sample(model,args,ex_file, datadirs=""):


    if not os.path.isdir(datadirs + '_Samples'):
        os.makedirs(datadirs+'_Samples', exist_ok=True)
    ex_file=datadirs+'_Samples/'+ex_file.split('.')[0]+'.npy'

    X=[]
    for i in np.int32(np.arange(0,args.num_sample,model.bsz)):
        x = model.sample_from_z_prior()
        X += [x.detach().cpu().numpy()]

    X=np.uint8(np.concatenate(X,axis=0)*255)
    np.save(ex_file,X)

    return


def make_images(test,model,ex_file,args, datadirs=""):

    if (True):

        if not os.path.isdir(datadirs+'_Images'):
            os.makedirs(datadirs+'_Images', exist_ok=True)
        ex_f=datadirs+'_Images/'+ex_file.split('.')[0]
        old_bsz=model.bsz
        model.bsz = 100
        num_mu_iter=None
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if (args.n_class):
            for c in range(model.n_class):
                ind=(test[1]==c)
                show_reconstructed_images([test[0][ind]],model,ex_f,args.nti,c, args.erode)
        else:
            show_reconstructed_images(test,model,ex_f,args.nti,None, args.erode)

        if args.n_mix>1:
            for clust in range(args.n_mix):
                show_sampled_images(model,ex_f,clust)
        else:
            show_sampled_images(model, ex_f)




        model.bsz=old_bsz


def create_img(XX,c,h,w,ri=10,rj=10,sep=0):
    mat = []
    t = 0
    for i in range(ri):
        line = []
        for j in range(rj):
            if (t < len(XX)):
                line += [XX[t].reshape((c, h, w)).transpose(1, 2, 0)]
            else:
                line += [np.zeros((c,h,w)).transpose(1, 2, 0)]
            line+=[np.ones((c,sep,w)).transpose(1,2,0)]
            t += 1
        mat += [np.concatenate(line, axis=0)]
    manifold = np.concatenate(mat, axis=1)
    if (c==1):
        img = np.concatenate([manifold, manifold, manifold], axis=2)
    else:
        img=manifold
    return img

def create_image(XX, model, ex_file):

    img=create_img(XX,model.input_channels,model.h,model.w)

    imsave(ex_file+'.png', np.uint8(img*255))

    #print("Saved the sampled images")

def show_sampled_images(model,ex_file,clust=None):
    theta = torch.zeros(model.bsz, model.u_dim)
    X=model.sample_from_z_prior(theta,clust)
    XX=X.detach().cpu().numpy()
    if clust is not None:
        ex_file=ex_file+'_'+str(clust)
    create_image(XX, model, ex_file)


def show_reconstructed_images(test,model,ex_file,num_iter=None, cl=None, erd=False):

    np.random.shuffle(test[0])
    inp=torch.from_numpy(erode(erd,test[0][0:100]))


    if (cl is not None):
        X,_,_,_=model.recon(inp,num_iter,cl)
    else:
        X,_,_,_ = model.recon(inp, num_iter)
    X = X.cpu().detach().numpy().reshape(inp.shape)
    XX=np.concatenate([inp[0:50],X[0:50]])
    if (cl is not None):
        create_image(XX,model, ex_file+'_recon'+'_'+str(cl))
    else:
        create_image(XX, model, ex_file + '_recon')

def add_occlusion(recon_data):
    recon_data[0][0:20,0:13,:,:]=0
    return recon_data

def add_clutter(recon_data):

    block_size=3
    num_clutter=2
    dim=np.zeros((1,2))
    dim[0,0]=recon_data[0].shape[1]-block_size
    dim[0,1]=recon_data[0].shape[2]-block_size
    qq=np.random.rand(recon_data.shape[0],num_clutter,2)
    rr=np.int32(qq*dim)
    for  rrr,im in zip(rr,recon_data):
        for k in range(num_clutter):
            x=rrr[k,0]
            y=rrr[k,1]
            im[0,x:x+block_size,y:y+block_size]=np.ones((block_size,block_size))

    return recon_data



def erode(do_er,data):

    #rdata=rotate_dataset_rand(data) #,angle=40,scale=.2)
    rdata=data
    if (do_er):
        el=np.zeros((3,3))
        el[0,1]=el[1,0]=el[1,2]=el[2,1]=el[1,1]=1
        rr=np.random.rand(len(data))<.5
        ndata=np.zeros_like(data)
        for r,ndd,dd in zip(rr,ndata,rdata):
            if (r):
                dda=ndimage.binary_erosion(dd[0,:,:]>0,el).astype(dd.dtype)
            else:
                dda=ndimage.binary_dilation(dd[0,:,:]>.9,el).astype(dd.dtype)
            ndd[0,:,:]=dda
    else:
        ndata=rdata

    return ndata



def deform_data(x_in,dv,perturb,trans,s_factor,h_factor,embedd):
        h=x_in.shape[2]
        w=x_in.shape[3]
        nn=x_in.shape[0]
        v=((torch.rand(nn, 6) - .5) * perturb)
        rr = torch.zeros(nn, 6)
        if not embedd:
            ii = torch.randperm(nn)
            u = torch.zeros(nn, 6)
            u[ii[0:nn//2]]=v[ii[0:nn//2]]
        else:
            u=v
        # Ammplify the shift part of the
        u[:,[2,5]]*=2
        rr[:, [0,4]] = 1
        if trans=='shift':
          u[:,[0,1,3,4]]=0
        elif trans=='scale':
          u[:,[1,3]]=0
        elif 'rotate' in trans:
          u[:,[0,1,3,4]]*=1.5
          ang=u[:,0]
          v=torch.zeros(nn,6)
          v[:,0]=torch.cos(ang)
          v[:,1]=-torch.sin(ang)
          v[:,4]=torch.cos(ang)
          v[:,3]=torch.sin(ang)
          s=torch.ones(nn)
          if 'scale' in trans:
            s = torch.exp(u[:, 1])
          u[:,[0,1,3,4]]=v[:,[0,1,3,4]]*s.reshape(-1,1).expand(nn,4)
          rr[:,[0,4]]=0
        theta = (u+rr).view(-1, 2, 3)
        grid = F.affine_grid(theta, [nn,1,h,w],align_corners=True)
        x_out=F.grid_sample(x_in,grid,padding_mode='zeros',align_corners=True)

        if x_in.shape[1]==3 and s_factor>0:
            v=torch.rand(nn,2)
            vv=torch.pow(2,(v[:,0]*s_factor-s_factor/2)).reshape(nn,1,1)
            uu=((v[:,1]-.5)*h_factor).reshape(nn,1,1)
            x_out_hsv=rgb_to_hsv(x_out)
            x_out_hsv[:,1,:,:]=torch.clamp(x_out_hsv[:,1,:,:]*vv,0.,1.)
            x_out_hsv[:,0,:,:]=torch.remainder(x_out_hsv[:,0,:,:]+uu,1.)
            x_out=hsv_to_rgb(x_out_hsv)

        ii=torch.where(torch.bernoulli(torch.ones(nn)*.5)==1)
        for i in ii:
              x_out[i]=x_out[i].flip(3)
        return x_out

def rgb_to_hsv(input):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    mx, inmx = torch.max(input, dim=1)
    mn, inmc = torch.min(input, dim=1)
    df = mx - mn
    h = torch.zeros(input.shape[0], 1)
    # if False: #'xla' not in device.type:
    #     h.to(device)
    ii = [0, 1, 2]
    iid = [[1, 2], [2, 0], [0, 1]]
    shift = [360, 120, 240]

    for i, id, s in zip(ii, iid, shift):
        logi = (df != 0) & (inmx == i)
        h[logi, 0] = \
            torch.remainder((60 * (input[logi, id[0]] - input[logi, id[1]]) / df[logi] + s), 360)

    s = torch.zeros(input.shape[0], 1) #
    # if False: #'xla' not in device.type:
    #     s.to(device)
    s[mx != 0, 0] = (df[mx != 0] / mx[mx != 0]) * 100

    v = mx.reshape(input.shape[0], 1) * 100

    output = torch.cat((h / 360., s / 100., v / 100.), dim=1)

    output = output.reshape(sh).transpose(1, 3)
    return output

def hsv_to_rgb(input):
    input = input.transpose(1, 3)
    sh = input.shape
    input = input.reshape(-1, 3)

    hh = input[:, 0]
    hh = hh * 6
    ihh = torch.floor(hh).type(torch.int32)
    ff = (hh - ihh)[:, None];
    v = input[:, 2][:, None]
    s = input[:, 1][:, None]
    p = v * (1.0 - s)
    q = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)));

    output = torch.zeros_like(input) #.to(device)
    # if False: #'xla' not in device.type:
    #     output.to(device)
    output[ihh == 0, :] = torch.cat((v[ihh == 0], t[ihh == 0], p[ihh == 0]), dim=1)
    output[ihh == 1, :] = torch.cat((q[ihh == 1], v[ihh == 1], p[ihh == 1]), dim=1)
    output[ihh == 2, :] = torch.cat((p[ihh == 2], v[ihh == 2], t[ihh == 2]), dim=1)
    output[ihh == 3, :] = torch.cat((p[ihh == 3], q[ihh == 3], v[ihh == 3]), dim=1)
    output[ihh == 4, :] = torch.cat((t[ihh == 4], p[ihh == 4], v[ihh == 4]), dim=1)
    output[ihh == 5, :] = torch.cat((v[ihh == 5], p[ihh == 5], q[ihh == 5]), dim=1)

    output = output.reshape(sh)
    output = output.transpose(1, 3)
    return output


class Edge(torch.nn.Module):
    def __init__(self, device, ntr=4, dtr=0):
        super(Edge, self).__init__()
        self.ntr = ntr
        self.dtr = dtr
        self.dv = device

    def forward(self, x):
        x = self.pre_edges(x).to(self.dv)
        return x


    def pre_edges(self, im):

        with torch.no_grad():
            EDGES=[]
            for k in range(im.shape[1]):
                EDGES+=[self.get_edges(im[:,k,:,:])]

            ED=torch.cat(EDGES,dim=1)

        return ED

    def get_edges(self,im):

        sh=im.shape
        delta=3
        im_b=torch.ones((sh[0],sh[1]+2*delta,sh[2]+2*delta)).to(self.dv)
        im_b[:,delta:delta+sh[1],delta:delta+sh[2]]=im

        diff_11 = torch.roll(im_b,(1,1),dims=(1,2))-im_b
        diff_nn11 = torch.roll(im_b, (-1, -1) ,dims=(1,2)) - im_b

        diff_01 = torch.roll(im_b,(0,1), dims=(1,2))-im_b
        diff_n01 = torch.roll(im_b,(0,-1),dims=(1,2))-im_b
        diff_10 = torch.roll(im_b,(1,0), dims=(1,2))-im_b
        diff_n10 = torch.roll(im_b,(-1,0),dims=(1,2))-im_b
        diff_n11 = torch.roll(im_b,(-1,1),dims=(1,2))-im_b
        diff_1n1 = torch.roll(im_b,(1,-1),dims=(1,2))-im_b

        thresh=self.ntr
        dtr=self.dtr
        ad_10=torch.abs(diff_10)
        ad_10=ad_10*(ad_10>dtr).float()
        e10a=torch.gt(ad_10,torch.abs(diff_01)).type(torch.uint8)\
             + torch.gt(ad_10,torch.abs(diff_n01)).type(torch.uint8) + torch.gt(ad_10,torch.abs(diff_n10)).type(torch.uint8)
        e10b=torch.gt(ad_10,torch.abs(torch.roll(diff_01,(1,0),dims=(1,2)))).type(torch.uint8)+\
                    torch.gt(ad_10, torch.abs(torch.roll(diff_n01, (1, 0), dims=(1, 2)))).type(torch.uint8)+\
                            torch.gt(ad_10,torch.abs(torch.roll(diff_01, (1, 0), dims=(1, 2)))).type(torch.uint8)
        e10 = torch.gt(e10a+e10b,thresh) & (diff_10>0)
        e10n =torch.gt(e10a+e10b,thresh) & (diff_10<0)

        ad_01 = torch.abs(diff_01)
        ad_01 = ad_01*(ad_01>dtr).float()
        e01a = torch.gt(ad_01, torch.abs(diff_10)).type(torch.uint8) \
               + torch.gt(ad_01, torch.abs(diff_n10)).type(torch.uint8) + torch.gt(ad_01, torch.abs(diff_n01)).type(torch.uint8)
        e01b = torch.gt(ad_01, torch.abs(torch.roll(diff_10, (0, 1), dims=(1, 2)))).type(torch.uint8) + \
                torch.gt(ad_01, torch.abs(torch.roll(diff_n10, (0, 1), dims=(1, 2)))).type(torch.uint8) +\
                    torch.gt(ad_01, torch.abs(torch.roll(diff_01, (0, 1), dims=(1, 2)))).type(torch.uint8)
        e01 = torch.gt(e01a + e01b, thresh) & (diff_01 > 0)
        e01n = torch.gt(e01a + e01b, thresh) & (diff_01 < 0)

        ad_11 = torch.abs(diff_11)
        ad_11 = ad_11*(ad_11>dtr).float()
        e11a = torch.gt(ad_11, torch.abs(diff_n11)).type(torch.uint8) \
               + torch.gt(ad_11, torch.abs(diff_1n1)).type(torch.uint8) + torch.gt(ad_11, torch.abs(diff_nn11)).type(torch.uint8)
        e11b = torch.gt(ad_11, torch.abs(torch.roll(diff_n11, (1, 1), dims=(1, 2)))).type(torch.uint8) + \
                torch.gt(ad_11, torch.abs(torch.roll(diff_1n1, (1, 1), dims=(1, 2)))).type(torch.uint8) + \
                    torch.gt(ad_11, torch.abs(torch.roll(diff_11, (1, 1), dims=(1, 2)))).type(torch.uint8)
        e11 = torch.gt(e11a + e11b, thresh) & (diff_11 > 0)
        e11n = torch.gt(e11a + e11b , thresh) & (diff_11 < 0)

        ad_n11 = torch.abs(diff_n11)
        ad_n11 = ad_n11*(ad_n11>dtr).float()
        en11a = torch.gt(ad_n11, torch.abs(diff_11)).type(torch.uint8) \
               + torch.gt(ad_n11, torch.abs(diff_1n1)).type(torch.uint8) + torch.gt(ad_n11, torch.abs(diff_nn11)).type(torch.uint8)
        en11b = torch.gt(ad_n11, torch.abs(torch.roll(diff_11, (-1, 1), dims=(1, 2)))).type(torch.uint8) + \
               torch.gt(ad_n11, torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.uint8) + \
               torch.gt(ad_n11, torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.uint8)
        en11 = torch.gt(en11a + en11b, thresh) & (diff_n11 > 0)
        en11n = torch.gt(en11a + en11b, thresh) & (diff_n11 < 0)

        edges=torch.zeros((im.shape[0],8,im.shape[1],im.shape[2])).to(self.dv)
        edges[:,0,2:sh[1],0:sh[2]]=e10[:,delta+2:delta+sh[1],delta:delta+sh[2]]
        edges[:,1,0:sh[1]-2,0:sh[2]]=e10n[:,delta:delta+sh[1]-2,delta:delta+sh[2]]
        edges[:,2,0:sh[1], 2:sh[2]] = e01[:, delta:delta + sh[1], delta+2:delta + sh[2]]
        edges[:,3,0:sh[1], 0:sh[2]-2] = e01n[:, delta:delta + sh[1], delta:delta + sh[2]-2]
        edges[:,4,2:sh[1], 2:sh[2]] = e11[:, delta + 2:delta + sh[1], delta+2:delta + sh[2]]
        edges[:,5,0:sh[1] - 2, 0:sh[2]-2] = e11n[:, delta:delta + sh[1] - 2, delta:delta + sh[2]-2]
        edges[:,6,0:sh[1]-2, 2:sh[2]] = en11[:, delta:delta + sh[1]-2, delta+2:delta + sh[2]]
        edges[:,7,2:sh[1], 0:sh[2]-2] = en11n[:, delta+2:delta + sh[1], delta:delta + sh[2]-2]

        return(edges)

