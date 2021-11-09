import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import time
from mix import STVAE_mix

import contextlib
@contextlib.contextmanager
def dummy_context_mgr():
    yield None



class STVAE_mix_by_class(STVAE_mix):

    def __init__(self, sh, device, args):
        super(STVAE_mix_by_class, self).__init__(sh, device, args)

        self.n_class=args.n_class
        self.n_mix_perclass=int(self.n_mix/self.n_class)

        self.mu_lr = args.mu_lr
        self.eyy = torch.eye(self.n_mix).to(self.dv)


    def run_epoch_classify(self, train, d_type, fout=None, num_mu_iter=None, conf_thresh=0):


        self.eval()
        if self.opt:
            mu, logvar, ppi = self.initialize_mus(train[0], self.s_dim, True)
            mu = mu.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0, 1)
            logvar = logvar.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0, 1)
            ppi = ppi.reshape(-1, self.n_class, self.n_mix_perclass).transpose(0, 1)

        # ii = np.arange(0, train[0].shape[0], 1)
        # tr = train[0][ii]
        # y = train[1][ii]

        acc=0
        accb=0
        DF=[]; RY=[]; YY=[]
        for j in np.arange(0, train.num, train.batch_size):
            KD = []
            BB = []
            fout.write('Batch '+str(j)+'\n')
            fout.flush()
            bb=next(iter(train))
            data = bb[0].to(self.dv)
            y=bb[1].numpy()
            YY+=[y]
            #data = self.preprocess(data_in)
            data_d = data.detach()
            if (len(data)<self.bsz):
                self.setup_id(len(data))
            if self.opt:
                for c in range(self.n_class):
                    #t1=time.time()
                    rng = range(c * self.n_mix_perclass, (c + 1) * self.n_mix_perclass)
                    self.update_s(mu[c][j:j + self.bsz], logvar[c][j:j + self.bsz], ppi[c][j:j + self.bsz], self.mu_lr[0])

                    for it in range(num_mu_iter):
                            self.compute_loss_and_grad(data_d, data_in, None, 'test', self.optimizer_s, opt='mu',rng=rng)
                    if (self.s_dim==1):
                        ss_mu=torch.ones(self.mu.shape[0],self.n_mix_perclass,self.s_dim).transpose(0,1).to(self.dv)
                    else:
                        ss_mu = self.mu.reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)
                    pi = torch.softmax(self.pi, dim=1)
                    lpi=torch.log(pi)
                    with torch.no_grad():
                        recon_batch = self.decoder_and_trans(ss_mu, rng)
                        b=self.mixed_loss_pre(recon_batch, data)
                        B = torch.sum(pi * b, dim=1)
                    BB += [B]
                    KD += [self.dens_apply(self.mu, self.logvar, lpi, pi)[1]]
            else:
                with torch.no_grad():
                    s_mu, s_var, pi,_ = self.encoder_mix(data)

                    if (self.s_dim==1):
                        ss_mu=torch.ones(s_mu.shape[0],self.n_mix,self.s_dim).transpose(0,1).to(self.dv)
                    else:
                        ss_mu = s_mu.reshape(-1, self.n_mix, self.s_dim).transpose(0,1)
                    recon_batch = self.decoder_and_trans(ss_mu)
                    b = self.mixed_loss_pre(recon_batch, data)
                    b = b.reshape(-1,self.n_class,self.n_mix_perclass)
                    s_mu = s_mu.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0,1)
                    s_var = s_var.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0,1)
                    tpi=pi.reshape(-1,self.n_class,self.n_mix_perclass).transpose(0,1)

                for c in range(self.n_class):
                        pic=tpi[c]/torch.sum(tpi[c],dim=1).unsqueeze(1)
                        lpic=torch.log(pic)
                        KD += [self.dens_apply(s_mu[c], s_var[c], lpic, pic)[1]]
                        BB += [torch.sum(pic*b[:,c,:],dim=1)]

            KD=torch.stack(KD,dim=1)
            BB=torch.stack(BB, dim=1)
            rr = BB + KD
            vy, ry = torch.min(rr, 1)
            ry = np.int32(ry.detach().cpu().numpy())
            RY+=[ry]
            rr=rr.detach().cpu().numpy()
            ii=np.argsort(rr,axis=1)
            DF+=[np.diff(np.take_along_axis(rr, ii[:, 0:2], axis=1), axis=1)]
            acc += np.sum(np.equal(ry, y))
            acc_temp = acc/(j+len(data))
            fout.write('====> Epoch {}: Accuracy: {:.4f}\n'.format(d_type, acc_temp))
            fout.flush()
            #accb += np.sum(np.equal(by, y[j:j + self.bsz]))
        YY=np.concatenate(YY)
        RY=np.concatenate(RY)
        DF=np.concatenate(DF,axis=0)
        iip = DF[:,0]>=conf_thresh
        iid = np.logical_not(iip)
        cl_rate=np.sum(np.equal(RY[iip],YY[iip]))
        acc/=train.num
        fout.write('====> Epoch {}: Accuracy: {:.4f}\n'.format(d_type,acc))
        return(iid,RY,cl_rate,acc)

    def recon(self,input,num_mu_iter,cl, lower=False, back_ground=None):


        enc_out=None
        sdim=self.s_dim
        if not lower:
            self.lower=False
            sdim=self.s_dim
        else:
            self.lower=True
            sdim = self.decoder_m.z2h._modules['0'].lin.out_features

        if self.opt:
            mu, logvar, ppi = self.initialize_mus(input, sdim, True)
            mu=mu.reshape(-1,self.n_class,self.n_mix_perclass*sdim).transpose(0,1)
            logvar=logvar.reshape(-1,self.n_class,self.n_mix_perclass*sdim).transpose(0,1)
            ppi=ppi.reshape(-1,self.n_class,self.n_mix_perclass).transpose(0,1)

        num_inp=input.shape[0]
        #self.setup_id(num_inp)

        inp = input.to(self.dv)
        #inp = self.preprocess(input)

        c = cl
        rng = range(c * self.n_mix_perclass, (c + 1) * self.n_mix_perclass)

        if self.opt:
                self.update_s(mu[c], logvar[c], ppi[c], self.mu_lr[0])
                for it in range(num_mu_iter):
                    inp_d = inp.detach()
                    self.compute_loss_and_grad(inp_d,input, None, 'test', self.optimizer_s, opt='mu',rng=rng, back_ground=back_ground)
                if (self.s_dim == 1):
                    s_mu = torch.ones(self.mu.shape[0], self.n_mix_perclass, self.s_dim).transpose(0, 1).to(self.dv)
                else:
                    s_mu = self.mu.reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)
                pi = torch.softmax(self.pi, dim=1)
        else:
            with torch.no_grad():
                s_mu, s_var, pi, enc_out = self.encoder_mix(inp)
            if (self.s_dim == 1):
                s_mu = torch.ones(s_mu.shape[0], self.n_class, self.n_mix_perclass*self.s_dim).transpose(0, 1).to(self.dv)
                s_var = torch.ones(s_var.shape[0], self.n_class, self.n_mix_perclass*self.s_dim).transpose(0, 1).to(self.dv)
            else:
                s_mu = s_mu.reshape(-1, self.n_class, self.n_mix_perclass*self.s_dim).transpose(0,1)
                s_var = s_var.reshape(-1, self.n_class, self.n_mix_perclass*self.s_dim).transpose(0,1)


            pi = pi.reshape(-1, self.n_class, self.n_mix_perclass).transpose(0, 1)
            pi = pi[cl]

            s_mu = s_mu[cl].reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)
        with torch.no_grad():
            recon_batch = self.decoder_and_trans(s_mu,rng)
            if back_ground is not None:
                recon_batch = recon_batch * (recon_batch >= .2) + back_ground * (recon_batch < .2)

        recon_batch = recon_batch.transpose(0, 1)
        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix_perclass
        recon=recon_batch.reshape(self.n_mix_perclass*num_inp,-1)
        rr=recon[kk]


        return rr, None, None, None






