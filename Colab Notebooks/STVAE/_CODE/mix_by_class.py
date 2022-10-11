import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import time
from mix import STVAE_mix
from mix import initialize_mus, dens_apply
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
            mu, logvar, ppi = initialize_mus(train[0], self.s_dim, True)
            mu = mu.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0, 1)
            logvar = logvar.reshape(-1, self.n_class, self.n_mix_perclass * self.s_dim).transpose(0, 1)
            ppi = ppi.reshape(-1, self.n_class, self.n_mix_perclass).transpose(0, 1)

        # ii = np.arange(0, train[0].shape[0], 1)
        # tr = train[0][ii]
        # y = train[1][ii]

        acc=0
        accb=0
        DF=[]; RY=[]; YY=[]
        tra=iter(train)
        for j in np.arange(0, train.num, train.batch_size):
            KD = []
            BB = []
            fout.write('Batch '+str(j)+'\n')
            fout.flush()
            bb=next(tra)
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

    def update_ss(self, muu, mus, logvar, pi, mu_lr, prop=None, both=True):

        var = {}

        var['muu'] = torch.autograd.Variable(muu.to(self.dv), requires_grad=True)
        var['mus'] = torch.autograd.Variable(mus.to(self.dv), requires_grad=True)
        var['logvar'] = torch.autograd.Variable(logvar.to(self.dv), requires_grad=True)
        var['pi'] = torch.autograd.Variable(pi.to(self.dv), requires_grad=True)
        if prop is not None:
            var['prop'] = torch.autograd.Variable(prop.to(self.dv), requires_grad=True)

        PP1s = [var['mus'], var['logvar'], var['pi']]
        if prop is not None:
            PP1s+=[var['prop']]
        PP1u = [var['muu']]

        PP2 = []
        if both:
            for p in self.parameters():
                PP2 += [p]

        if self.optimizer_type == 'Adam':
            self.optimizer_s = optim.Adam([{'params': PP1u, 'lr': mu_lr},
                                           {'params': PP1s, 'lr': mu_lr / 10.},
                                           {'params': PP2}], lr=self.lr)
        else:
            self.optimizer_s = optim.SGD([{'params': PP1u, 'lr': mu_lr},
                                          {'params': PP1s, 'lr': mu_lr / 10.},
                                          {'params': PP2}], lr=self.lr)

        return var


    def recon(self,args, input,num_mu_iter,cl, lower=False, back_ground=None):


        enc_out=None
        sdim=self.s_dim
        if not lower:
            self.lower=False
            shape=self.final_shape
        else:
            self.lower=True
            shape = self.decoder_m.dec_conv_bot.temp.input_shape

        if self.opt:
            mu, logvar, ppi = initialize_mus(input.shape[0], shape, self.n_mix)
            mu=mu.reshape(-1,self.n_class,self.n_mix_perclass*sdim).transpose(0,1)
            logvar=logvar.reshape(-1,self.n_class,self.n_mix_perclass*sdim).transpose(0,1)
            ppi=ppi.reshape(-1,self.n_class,self.n_mix_perclass).transpose(0,1)

        num_inp=input.shape[0]
        #self.setup_id(num_inp)

        inp = input.to(self.dv)
        inp_d = inp.detach()
        #inp = self.preprocess(input)

        c = cl
        rng = range(c * self.n_mix_perclass, (c + 1) * self.n_mix_perclass)

        if self.opt:
                prop=None
                lrr=self.mu_lr[0]
                if back_ground is not None:
                    lrr=self.mu_lr[1]
                    prop=-2.*torch.ones([num_inp]+list(self.initial_shape))
                lrr = self.mu_lr[0]
                var=self.update_s(mu[c][:,:self.u_dim], mu[c][:,self.u_dim:],logvar[c], ppi[c], lrr, prop=prop, both=self.nosep)
                for it in range(num_mu_iter):

                    rls,ls,_,pmix=self.compute_loss_and_grad(var, inp_d,inp, None, 'test', self.optimizer_s, opt='mu',rng=rng, back_ground=back_ground)
                s_mu = torch.cat((var['muu'],var['mus']),dim=1)#.reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)
                pi=var['pi']
        else:
            var={}
            with torch.no_grad():
                var, enc_out = self.encoder_mix(inp)
                s_mu = var['mu'].reshape(-1, self.n_class, self.n_mix_perclass*self.s_dim).transpose(0,1)

            pi = var['pi'].reshape(-1, self.n_class, self.n_mix_perclass).transpose(0, 1)
            pi = pi[cl]
            s_mu = s_mu[cl].reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)

        recon_batch_both = None
        with torch.no_grad():
            if self.opt:
                var['pi'] = pi
                var['mu'] = s_mu
                if back_ground is not None :
                    recloss, totloss, recon_batch_both, pmix = self.get_loss_background(inp_d, var, back_ground,rng=rng)
                else:
                    recloss, totloss, recon_batch,_ = self.get_loss(inp_d,  None, var, rng=rng, back_ground=None)

            ss_mu = s_mu.reshape(-1, self.n_mix_perclass, self.s_dim).transpose(0, 1)
            recon_batch = self.decoder_and_trans(ss_mu,rng)
            pi=torch.softmax(pi,dim=1)
            recloss = self.mixed_loss(recon_batch, inp_d, pi)
            totloss=0
            if back_ground is None and self.type != 'ae' and not self.opt:
                totloss = dens_apply(self.rho[cl], var['mu'], var['logvar'], torch.log(pi), pi)[0]
        recon_batch = recon_batch.transpose(0, 1)
        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix_perclass
        recon=recon_batch.reshape(self.n_mix_perclass*num_inp,-1)
        rr=recon[kk]
        print('recloss', recloss / num_inp, 'totloss', totloss / num_inp)

        return rr, None, None, None, recon_batch, recon_batch_both






