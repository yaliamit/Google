import torch
import torch.nn.functional as F
from torch import nn, optim
from images import deform_data
import numpy as np
from tps import TPSGridGen
from encoder_decoder import encoder_mix, decoder_mix
from enc_conv import ENC_DEC
import contextlib
import time

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

def setup_trans_stuff(self,args):
        self.tf = args.transformation
        self.type=args.type
        if 'tvae' in self.type:
            if self.tf == 'aff':
                self.u_dim = 6
                self.idty = torch.cat((torch.eye(2),torch.zeros(2).unsqueeze(1)),dim=1).to(self.dv)
            elif self.tf == 'tps':
                self.u_dim = self.tps_num*self.tps_num*2 # 2 * 3 ** 2
                self.gridGen = TPSGridGen(out_h=self.h,out_w=self.w,grid_size=self.tps_num,device=self.dv)
                px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
                self.idty = torch.cat((px,py)).to(self.dv)
        else:
            self.u_dim=0


        self.z_dim = self.s_dim-self.u_dim

        return self



class STVAE_mix(nn.Module):


    def __init__(self, sh, device, args):
        super(STVAE_mix, self).__init__()
        self.bsz = args.mb_size
        self.dv = device
        self.binary_thresh=args.binary_thresh
        self.initial_shape=sh
        self.lim=args.lim
        self.h = sh[1]
        self.w = sh[2]
        self.opt = args.OPT

        self.opt_jump=args.opt_jump
        self.mu_lr = args.mu_lr
        self.lr=args.lr
        self.perturb=args.perturb
        self.s_dim = args.sdim
        self.n_mix = args.n_mix
        self.input_channels = sh[0]
        self.flag=True
        self.lower=False
        self.nosep=args.nosep
        self.lamda=args.lamda
        self.diag=args.Diag
        self.nti=args.nti
        self.optim=args.optimizer
        self.wd=args.wd
        self.n_class=args.n_class

        self.optimizer_type = args.optimizer
        self.decoder_nonlinearity = args.decoder_nonlinearity
        self.penalty=args.penalize_activations
        self.tps_num=args.tps_num
        if args.output_cont>0.:
            self.output_cont=nn.Parameter(torch.tensor(args.output_cont), requires_grad=False)
        else:
            self.output_cont=0.
        self.final_shape=None
        self.enc_conv=None
        setup_trans_stuff(self,args)
        if hasattr(args,'enc_layers') and args.enc_layers is not None:
            self.enc_conv=ENC_DEC(sh,self.dv,args)
            self.final_shape=np.array(self.enc_conv.model.output_shape)[1:]
            self.x_dim=np.prod(self.final_shape)
        else:
            self.x_dim=np.prod(sh)


        if (not args.OPT):
                self.encoder_m=encoder_mix(self,args)


        self.decoder_m=decoder_mix(self,args)


        self.rho = nn.Parameter(torch.zeros(self.n_mix))#,requires_grad=False)
        self.scheduler=None
        self.optimizer=None
        if (not args.nosep):
            if (args.optimizer=='Adam'):
                self.optimizer = optim.Adam(self.parameters())
            else:
                self.optimizer = optim.SGD(self.parameters(),lr=args.lr)

            #self.optimizer=optim.Adam(self.parameters(),lr=args.lr)
        #elif (args.optimizer=='Adadelta'):
        #   self.optimizer = optim.Adadelta(self.parameters())

    def encoder_mix(self,input):

        return self.encoder_m(input, self.enc_conv)

    def decoder_mix(self, input, rng=None, lower=False):

       if not lower:
            return self.decoder_m(input, self.enc_conv, rng=rng)
       else:
            x=self.decoder_m.h_to_x(input, self.enc_conv, rng=rng)
            return x,None
    def initialize_mus(self,len, s_dim,OPT=None):
        trMU=None
        trLOGVAR=None
        trPI=None
        sdim=s_dim
        if self.n_mix>0:
            sdim=sdim*self.n_mix
        if (len is not None):
            trMU = torch.zeros(len, sdim) #.to(self.dv)
            trLOGVAR = torch.zeros(len, sdim) #.to(self.dv)
            #EE = (torch.eye(self.n_mix) * 5. + torch.ones(self.n_mix)).to(self.dv)
            #ii=torch.randint(0,self.n_mix,[train.shape[0]])
            #trPI=EE[ii]
            trPI=torch.zeros(len, self.n_mix) #.to(self.dv)
        return trMU, trLOGVAR, trPI

    def update_s(self,mu,logvar,pi,mu_lr,wd=0, both=True):

        self.mu=torch.autograd.Variable(mu.to(self.dv), requires_grad=True)
        self.logvar = torch.autograd.Variable(logvar.to(self.dv), requires_grad=True)
        self.pi = torch.autograd.Variable(pi.to(self.dv), requires_grad=True)

        PP1=[self.mu, self.logvar,self.pi]
        PP2 = []
        if both:
            for p in self.parameters():
                PP2+=[p]

        if self.optim=='Adam':
            self.optimizer_s = optim.Adam([{'params':PP1,'lr':mu_lr},
                                      {'params':PP2}],lr=self.lr)
        else:
            self.optimizer_s = optim.SGD([{'params': PP1, 'lr': mu_lr},
                                           {'params': PP2}], lr=self.lr)



    def apply_trans(self,x,u):
        # Apply transformation
        if 'tvae' in self.type:
            #id = self.idty.expand((x.shape[0],) + self.idty.size()).to(self.dv)
            # Apply linear only to dedicated transformation part of sampled vector.
            if self.tf == 'aff':
                self.theta = u.view(-1, 2, 3) + self.idty.unsqueeze(0)
                grid = F.affine_grid(self.theta, x.view(-1,self.input_channels,self.h,self.w).size(),align_corners=True)
            elif self.tf=='tps':
                self.theta = u + self.idty.unsqueeze(0)
                grid = self.gridGen(self.theta)
            #elif self.tf == 'shifts':

            x = F.grid_sample(x.view(-1,self.input_channels,self.h,self.w), grid, padding_mode='border',align_corners=True)


        return x

    def decoder_and_trans(self,s, rng=None, train=True):

        n_mix=s.shape[0]

        x, u = self.decoder_mix(s,rng=rng, lower=self.lower)
        # Transform
        if (self.u_dim>0):
            xt = []
            for xx,uu in zip(x,u):
                    xt=xt+[self.apply_trans(xx,uu).squeeze()]

            x=torch.stack(xt,dim=0).reshape(n_mix,x.shape[1],-1)
        xx = torch.clamp(x, self.binary_thresh, 1 - self.binary_thresh)
        return xx


    def sample(self, mu, logvar, dim):
        #print(mu.shape,dim)
        eps = torch.randn(mu.shape[0],dim).to(self.dv)
        if (self.s_dim>1):
            z = mu + torch.exp(logvar/2) * eps
        else:
            z = torch.ones(mu.shape[0],dim).to(self.dv)
        return z

    def dens_apply(self,s_mu,s_logvar,lpi,pi):
        n_mix=pi.shape[1]
        s_mu = s_mu.reshape(s_mu.shape[0], n_mix, -1)
        s_logvar = s_logvar.reshape(s_logvar.shape[0], n_mix, -1)
        sd=torch.exp(s_logvar/2)
        var=sd*sd

        # Sum along last coordinate to get negative log density of each component.
        KD_dens=-0.5 * torch.sum(1 + s_logvar - s_mu ** 2 - var, dim=2) # +KL(N(\mu,\si)| N(0,1))
        KD_disc=lpi - F.log_softmax(self.rho,dim=0)#torch.log(torch.tensor(n_mix,dtype=torch.float)) # +KL(\pi,unif(1/n_mix))
        KD = torch.sum(pi * (KD_dens + KD_disc), dim=1)
        ENT=torch.sum(F.softmax(self.rho,dim=0)*F.log_softmax(self.rho,dim=0))
        tot=torch.sum(KD)+ENT

        return tot, KD, KD_dens

    def dens_apply_samp(self,s,s_mu, s_logvar,lpi,pi):

        s_lv=torch.sum(s_logvar.reshape(-1,self.n_mix,self.s_dim),dim=2)
        ll=torch.zeros(s.shape[1],s.shape[0]).to(self.dv)
        logds=torch.zeros(self.n_mix).to(self.dv)
        for k,zz in enumerate(self.decoder_m.z2z):
            e=torch.matmul(s[k,:,:]-zz.lin.bias,zz.lin.weight)
            e=e.squeeze()
            logds[k]=zz.logd #torch.logdet(torch.mm(zz.lin.weight.t(),zz.lin.weight))
            ll[:,k]= .5*torch.sum(e*e,dim=1)
        ll=ll-.5*s_lv-.5*self.n_mix
        KD_disc=lpi - F.log_softmax(self.rho, dim=0)
        tot = torch.sum(pi*(KD_disc+ll))
        ENT = torch.sum(F.softmax(self.rho, dim=0) * F.log_softmax(self.rho, dim=0))
        tot=tot+ENT-.5*torch.sum(pi*logds)
        return tot, KD_disc, ENT

    def mixed_loss_pre(self,x,data):
        b = []

        if (self.output_cont==0.):
            for xx in x:
                a = F.binary_cross_entropy(xx, data.reshape(data.shape[0], -1),
                                           reduction='none')
                a = torch.sum(a, dim=1)
                b = b + [a]
        else:
            for xx in x:
                datas=data.reshape(data.shape[0],-1)
                a=(datas-xx)*(datas-xx)
                a = torch.sum(a, dim=1)*self.output_cont
                b = b + [a]
        b = torch.stack(b).transpose(0, 1)
        return(b)

    def weighted_sum_of_likelihoods(self,lpi,b):
        return(-torch.logsumexp(lpi-b,dim=1))

    def mixed_loss(self,x,data,pi):

        b=self.mixed_loss_pre(x,data)
        recloss=torch.sum(pi*b)
        return recloss


    def get_loss(self,data_to_match,targ,mu,logvar,pi,rng=None, back_ground=None):

        #m = torch.rand(data.shape[0])<torch.tensor([1./self.n_class])

        if (targ is not None):
            pi=pi.reshape(-1,self.n_class,self.n_mix_perclass)
            pis=torch.sum(pi,2)
            pi = pi/pis.unsqueeze(2)
        lpi = torch.log(pi)
        n_mix = self.n_mix
        if (targ is None and self.n_class > 0):
            n_mix = self.n_mix_perclass
        if (self.type != 'ae'):
            s = self.sample(mu, logvar, mu.shape[1])
        else:
            s=mu
        s=s.reshape(data_to_match.shape[0],n_mix,-1).transpose(0,1)
        # Apply linear map to entire sampled vector.
        x=self.decoder_and_trans(s,rng)
        if back_ground is not None:
            x=x*(x>=.5)+back_ground*(x<.5)
        if (targ is not None):
            x=x.transpose(0,1)
            x=x.reshape(-1,self.n_class,self.n_mix_perclass,x.shape[-1])
            mu=mu.reshape(-1,self.n_class,self.n_mix_perclass*self.s_dim)
            logvar=logvar.reshape(-1,self.n_class,self.n_mix_perclass*self.s_dim)
            tot=0
            recloss=0
            if (type(targ) == torch.Tensor):
                for c in range(self.n_class):
                    ind = (targ == c)
                    tot += self.dens_apply(mu[ind,c,:],logvar[ind,c,:],lpi[ind,c,:],pi[ind,c,:])[0]
                    recloss+=self.mixed_loss(x[ind,c,:,:].transpose(0,1),data_to_match[ind],pi[ind,c,:])
            else:
                 tot += self.dens_apply(mu[:, targ, :], logvar[:, targ, :], lpi[:,targ,:], pi[:,targ,:])[0]
                 recloss += self.mixed_loss(x[:, targ, :, :].transpose(0, 1), data_to_match, pi[:, targ, :])
        else:
            tot=0.
            if (self.type != 'ae'):
                    tot, _, _ = self.dens_apply(mu, logvar, lpi, pi)
            recloss = self.mixed_loss(x, data_to_match, pi)
        return recloss, tot

    def encoder_and_loss(self, data,data_to_match, targ, rng,back_ground=None):

        with torch.no_grad() if not self.flag else dummy_context_mgr():
            if (self.opt):
                pi = torch.softmax(self.pi, dim=1)
                logvar=self.logvar
                mu=self.mu
            else:
                mu, logvar, pi, _ = self.encoder_mix(data)

        return self.get_loss(data_to_match,targ, mu,logvar,pi,rng,back_ground=back_ground)


    def compute_loss_and_grad(self,data,data_to_match,targ,d_type,optim, opt='par', rng=None,back_ground=None):

        optim.zero_grad()

        recloss, tot = self.encoder_and_loss(data,data_to_match,targ,rng,back_ground=back_ground)
        loss = recloss + tot

        if (d_type == 'train' or opt=='mu'):

            loss.backward(retain_graph=(opt=='mu'))

            optim.step()

        return recloss.item(), loss.item()



    def get_logdets(self):
            return

    def run_epoch(self, train, epoch,num_mu_iter, MU, LOGVAR,PI, d_type='test',fout=None):


        if (d_type=='train'):
            self.train()
        else:
            self.eval()

        tr_recon_loss = 0;tr_full_loss = 0
        ii = np.arange(0, train[0].shape[0], 1)
        # if (d_type=='train'):
        #   np.random.shuffle(ii)
        tr = train[0][ii]
        y = train[1][ii]
        mu = MU[ii]
        logvar = LOGVAR[ii]
        pi = PI[ii]
        self.epoch=epoch
        #print('batch_size',self.bsz)
        for j in np.arange(0, len(y), self.bsz):
            data_in = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            data = torch.from_numpy(tr[j:j + self.bsz]).float().to(self.dv)
            if self.perturb > 0. and d_type == 'train' and not self.opt:
                with torch.no_grad():
                    data = deform_data(data, self.dv, self.perturb, self.trans, self.s_factor, self.h_factor, True)
            data_d = data.detach()
            target=None
            if (self.n_class>0):
                target = torch.from_numpy(y[j:j + self.bsz]).float().to(self.dv)
            if self.opt:
                self.update_s(mu[j:j + self.bsz, :], logvar[j:j + self.bsz, :], pi[j:j + self.bsz], self.mu_lr[0],both=self.nosep)
                if np.mod(epoch, self.opt_jump) == 0:
                  for it in range(num_mu_iter):
                    recon_loss,loss = self.compute_loss_and_grad(data_in, data_d, target, d_type, self.optimizer_s, opt='mu')

            if not self.opt or not self.nosep:
              with torch.no_grad() if (d_type != 'train') else dummy_context_mgr():
                    recon_loss, loss=self.compute_loss_and_grad(data_in, data, target,d_type,self.optimizer)

            if self.opt:
                mu[j:j + self.bsz] = self.mu.data
                logvar[j:j + self.bsz] = self.logvar.data
                del self.mu, self.logvar
            if self.opt:
                pi[j:j + self.bsz] = self.pi.data
                del self.pi
            tr_recon_loss += recon_loss
            tr_full_loss += loss


        if (True): #(np.mod(epoch, 10) == 9 or epoch == 0):
            fout.write('\n====> Epoch {}: {} Reconstruction loss: {:.4f}, Full loss: {:.4F}\n'.format(d_type,
        epoch, tr_recon_loss / len(tr), tr_full_loss/len(tr)))

        return mu, logvar, pi, [tr_full_loss/len(tr), tr_recon_loss / len(tr)]

    def recon(self,input,num_mu_iter=None, lower=False,back_ground=None):

        enc_out=None
        self.eval()
        sdim=self.s_dim
        if not lower:
            self.lower=False
            sdim=self.s_dim
        else:
            self.lower=True
            sdim=self.decoder_m.z2h._modules['0'].lin.out_features

        if self.opt:
            mu, logvar, ppi = self.initialize_mus(input, sdim, True)

        num_inp=input.shape[0]
        #self.setup_id(num_inp)
        inp = input.float().to(self.dv)
        inp_d=inp.detach()
        if self.opt:
            self.update_s(mu, logvar, ppi, self.mu_lr[0],both=self.nosep)
            self.get_logdets()
            #torch.autograd.set_detect_anomaly(True)
            for it in range(num_mu_iter):
                rls, ls = self.compute_loss_and_grad(inp_d,inp, None, 'test', self.optimizer_s, opt='mu', back_ground=back_ground)
                #print(rls,ls)
            s_mu = self.mu
            s_var = self.logvar
            pi = torch.softmax(self.pi, dim=1)
        else:
            s_mu, s_var, pi, enc_out = self.encoder_mix(inp)

        #s = self.sample(s_mu, s_var, self.s_dim * self.n_mix)
            # for it in range(num_mu_iter):
            #     self.compute_loss_and_grad_mu(inp_d, s_mu, s_var, None, 'test', self.optimizer_s,
            #                                   opt='mu')

        ss_mu = s_mu.reshape(-1, self.n_mix, self.s_dim).transpose(0,1)

        ii = torch.argmax(pi, dim=1)
        jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
        kk = ii+jj*self.n_mix
        lpi = torch.log(pi)
        with torch.no_grad():
            recon_batch = self.decoder_and_trans(ss_mu)
            #if back_ground is not None:
            #    recon_batch = recon_batch * (recon_batch >= .2) + back_ground * (recon_batch < .2)
            recloss = self.mixed_loss(recon_batch, inp, pi)
            totloss = self.dens_apply(s_mu, s_var, lpi, pi)[0]

        recon_batch = recon_batch.transpose(0, 1)
        recon=recon_batch.reshape(self.n_mix*num_inp,-1)
        rr=recon[kk]
        print('recloss',recloss/num_inp,'totloss',totloss/num_inp)
        return rr, torch.cat([s_mu, s_var, pi],dim=1),[recloss,totloss], enc_out



    def sample_from_z_prior(self,theta=None, clust=None, lower=False):


        self.eval()
        if not lower:
            s_dim=self.s_dim
        else:
            self.lower=True
            s_dim=self.decoder_m.z2h._modules['0'].lin.out_features
        rho_dist=torch.exp(self.rho-torch.logsumexp(self.rho,dim=0))
        if (clust is not None):
            ii=clust*torch.ones(self.bsz, dtype=torch.int64).to(self.dv)
        else:
            ii=torch.multinomial(rho_dist,self.bsz,replacement=True)
        if (s_dim>1):
            s = torch.randn(self.bsz, s_dim*self.n_mix).to(self.dv)
        else:
            s = torch.ones(self.bsz,s_dim*self.n_mix).to(self.dv)

        s = s.reshape(-1, self.n_mix, s_dim)
        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            for i in range(self.n_mix):
                s[:,i,0:self.u_dim]=theta
        s = s.transpose(0,1)
        with torch.no_grad():
            x=self.decoder_and_trans(s,train=False)
        if hasattr(self,'enc_conv'):
        #if (self.feats and not self.feats_back):
            rec_b = []
            for rc in x:
                rec_b += [rc.reshape([-1]+list(self.initial_shape))]
            x = torch.stack(rec_b, dim=0)

        x=x.transpose(0,1)
        jj = torch.arange(0, self.bsz, dtype=torch.int64).to(self.dv)
        kk = ii + jj * self.n_mix
        recon = x.reshape(self.n_mix * self.bsz, -1)
        rr = recon[kk]

        return rr



    def compute_likelihood(self,Input,num_samples):

       self.eval()
       Input = torch.from_numpy(Input)
       num_inp = Input.shape[0]


       bsz=np.maximum(1,np.int(self.bsz/num_samples))
       print(bsz)
       LLG=0
       EEE = torch.randn(num_samples, num_inp, self.n_mix, self.s_dim).to(self.dv)
       lsfrho=torch.log_softmax(self.rho, 0)
       lns=np.log(num_samples)
       for j in np.arange(0, num_inp, bsz):

            input = Input[j:j+bsz].to(self.dv)


            if self.opt:
                mu, logvar, ppi = self.initialize_mus(input, True)



            if self.opt:
                inp_d = input.detach()
                self.update_s(mu, logvar, ppi, self.mu_lr[0],both=self.nosep)
                #self.get_logdets()
                for it in range(self.nti):
                    self.compute_loss_and_grad(inp_d,input, None, 'test', self.optimizer_s, opt='mu')
                s_mu = self.mu
                s_var = self.logvar
                pi = torch.softmax(self.pi, dim=1)
            else:
                with torch.no_grad():
                    s_mu, s_var, pi, _ = self.encoder_mix(input)

            EE=EEE[:,j:j+bsz,:,:].to(self.dv)
            s_var=s_var.reshape(1,bsz,self.n_mix,self.s_dim)
            tsd=torch.exp(s_var/2)
            tmu=s_mu.reshape(1,bsz,self.n_mix,self.s_dim)
            S =tmu +tsd*EE

            with torch.no_grad():

                recon_batch=self.decoder_and_trans(S.reshape(-1, self.n_mix, self.s_dim).transpose(0, 1))
                inp=input.repeat(num_samples,1,1,1)
                recon_loss = -self.mixed_loss_pre(recon_batch, inp).reshape(num_samples,bsz,self.n_mix)

            logq = -torch.sum((S - tmu) * (S - tmu) / (2 * tsd*tsd),dim=3)\
                   - torch.sum(s_var,dim=3)/2 #+ torch.log(pi.unsqueeze(0))
            logp = -.5*torch.sum(S*S,dim=3)
            LG = torch.logsumexp(recon_loss + logp - logq,0)
            LLG -= torch.sum(torch.logsumexp(LG+lsfrho,1))/num_inp

       LLG += lns
       print('LLL',LLG)

       return(LLG)

    def get_scheduler(self,args):
        self.scheduler=None
        if args.sched==1.:
            l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)
        elif args.sched==2.:
            self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,verbose=True)

