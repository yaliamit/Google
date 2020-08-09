import torch
import numpy as np
import os
from imageio import imsave
from scipy import ndimage

def make_images(test,model,ex_file,args):

    if (True):
        old_bsz=model.bsz
        model.bsz = 100
        model.setup_id(model.bsz)
        num_mu_iter=None
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if args.n_mix>0:
            for clust in range(args.n_mix):
                show_sampled_images(model,ex_file,clust)
        else:
            show_sampled_images(model, ex_file)

        if (args.n_class):
            for c in range(model.n_class):
                ind=(test[1]==c)
                show_reconstructed_images([test[0][ind]],model,ex_file,args.nti,c, args.erode)
        else:
            show_reconstructed_images(test,model,ex_file,args.nti,None, args.erode)



        model.bsz=old_bsz
        model.setup_id(old_bsz)

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


    if not os.path.isdir('_Images'):
        os.system('mkdir _Images')
    imsave('_Images/'+ex_file+'.png', np.uint8(img*255))

    #print("Saved the sampled images")

def show_sampled_images(model,ex_file,clust=None):
    theta = torch.zeros(model.bsz, model.u_dim)
    X=model.sample_from_z_prior(theta,clust)
    XX=X.cpu().detach().numpy()
    if clust is not None:
        ex_file=ex_file+'_'+str(clust)
    create_image(XX, model, ex_file)


def show_reconstructed_images(test,model,ex_file,num_iter=None, cl=None, erd=False):

    inp=torch.from_numpy(erode(erd,test[0][0:100]))


    if (cl is not None):
        X=model.recon(inp,num_iter,cl)
    else:
        X,_,_ = model.recon(inp, num_iter)
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

def prepare_recons(model, DATA, args,fout):
    dat = []
    HV=[]
    tips=['train','val','test']
    rr=range(0,3)
    if args.rerun:
        rr=range(2,3)
    for k in rr:
        totloss = 0
        recloss = 0
        if (DATA[k][0] is not None):
            INP = torch.from_numpy(DATA[k][0])
            if k==0:
                INP = INP[0:args.network_num_train]
            RR = []
            HVARS=[]
            for j in np.arange(0, INP.shape[0], 500):
                inp = INP[j:j + 500]
                rr, h_vars, losses= model.recon(inp, args.nti)
                recloss+=losses[0]
                totloss+=losses[1]
                RR += [rr.detach().cpu().numpy()]
                HVARS += [h_vars.detach().cpu().numpy()]
            RR = np.concatenate(RR)
            HVARS = np.concatenate(HVARS)
            tr = RR.reshape(-1, model.input_channels,model.h,model.w)
            dat += [[tr, DATA[k][1][0:INP.shape[0]]]]
            HV+=[[HVARS,DATA[k][1][0:INP.shape[0]]]]
            if (k==2):
                fout.write('\n====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(tips[k],
                                                            0,recloss / INP.shape[0], (recloss+totloss)/INP.shape[0]))

            print(k,recloss/INP.shape[0],(totloss+recloss)/INP.shape[0])
        else:
            dat += [DATA[k]]
            HV += [DATA[k]]
    print("Hello")

    return dat, HV


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

