import torch
import torch.nn.functional as F
from torch import nn, optim
from images import deform_data, add_clutter
import numpy as np
from tps import TPSGridGen
from encoder_decoder import encoder_mix, decoder_mix, flow_mix
import contextlib
from network_aux import temp_args
from get_net_text import get_network
import copy
import time

@contextlib.contextmanager
def dummy_context_mgr():
    yield None

def initialize_mus(len, fshape, n_mix=0):
        trMU = None
        trLOGVAR = None
        trPI = None
        sh=copy.copy(fshape)
        if n_mix > 0:
            sh[0]*=n_mix
        if (len is not None):
            trMU = torch.zeros([len]+list(sh))
            trLOGVAR = torch.zeros([len]+list(sh))
            trPI = torch.zeros(len, n_mix)
        return trMU, trLOGVAR, trPI

class apply_trans(object):

    def __init__(self,args,sh,dv):
        self.h=sh[1]
        self.w=sh[2]
        self.input_channels=args.input_channels
        self.tf=args.transformation
        self.type=args.type
        self.dv=dv

        if self.tf=='shift':
            self.u_dim=2
            self.idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1).to(dv)
        elif self.tf=='aff':
            self.u_dim=6
            self.idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1).to(dv)
        else:
            self.u_dim = args.tps_num * args.tps_num * 2  # 2 * 3 ** 2
            self.gridGen = TPSGridGen(out_h=self.h, out_w=self.w, grid_size=args.tps_num, device=dv)
            px = self.gridGen.PX.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            py = self.gridGen.PY.squeeze(0).squeeze(0).squeeze(0).squeeze(0)
            self.idty = torch.cat((px, py)).to(self.dv)


    def __call__(self,x,u):

        if self.tf is not None:
            #id = self.idty.expand((x.shape[0],) + self.idty.size()).to(self.dv)
            # Apply linear only to dedicated transformation part of sampled vector.

            if self.tf=='shift':
                theta=torch.zeros(u.shape[0],2,3).to(self.dv) + self.idty.unsqueeze(0)
                theta[:,:,2]=u
                grid = F.affine_grid(theta, x.view(-1,self.input_channels,self.h,self.w).size(),align_corners=True)
            if self.tf == 'aff':
                theta = u.view(-1, 2, 3) + self.idty.unsqueeze(0)
                grid = F.affine_grid(theta, x.view(-1,self.input_channels,self.h,self.w).size(),align_corners=True)
            elif self.tf=='tps':
                theta = u + self.idty.unsqueeze(0)
                grid = self.gridGen(theta)
            x = F.grid_sample(x.view(-1,self.input_channels,self.h,self.w), grid, padding_mode='border',align_corners=True)


        return x

def setup_trans_stuff(self,args,sh,dv):
        self.tf = args.transformation
        self.type=args.type
        if self.tf is not None:
            self.apply_trans=apply_trans(args,sh,dv)
            self.u_dim = self.apply_trans.u_dim
            self.z_dim = args.sdim - self.apply_trans.u_dim
        else:
            self.u_dim=0
            self.z_dim=args.sdim



        return self

def dens_apply(rho,s_mu,s_logvar, lpi,pi):
        n_mix=pi.shape[1]
        sh=[s_mu.shape[0],n_mix,s_mu.shape[1]//n_mix]+list(s_mu.shape[2:])
        lensh=len(sh)
        s_mu = s_mu.reshape(sh)
        s_logvar = s_logvar.reshape(sh)

        sd=torch.exp(s_logvar/2)
        var=sd*sd

        # Sum along last coordinate to get negative log density of each component.
        KD_dens=-0.5 * torch.sum((1 + s_logvar - s_mu ** 2 - var), dim=list(range(2,lensh)))# +KL(N(\mu,\si)| N(0,1))
        KD_disc=lpi - F.log_softmax(rho,dim=0)#torch.log(torch.tensor(n_mix,dtype=torch.float)) # +KL(\pi,unif(1/n_mix))
        KD = torch.sum(pi * (KD_dens + KD_disc), dim=1)
        #ENT=torch.sum(F.softmax(rho,dim=0)*F.log_softmax(rho,dim=0))
        tot=torch.sum(KD) #+ENT

        return tot, KD, KD_dens

def setup_optimizer(sel,args):
            KEYS=[]
            for keys, vals in sel.named_parameters():
                if 'running' not in keys and 'tracked' not in keys:
                    KEYS+=[keys]

            pp = []
            totpars=0
            for k, p in zip(KEYS, sel.parameters()):
                if (args.update_layers is None):
                    args.fout.write('TO optimizer ' + k + str(np.array(p.shape)) + '\n')
                    totpars += np.prod(np.array(p.shape))
                    pp += [p]
                else:
                    found = False
                    for u in args.update_layers:
                        if u in k.split('.'): #u == k.split('.')[1] or u == k.split('.')[0]:
                            found = True
                            args.fout.write('TO optimizer ' + k + str(np.array(p.shape)) + '\n')
                            totpars += np.prod(np.array(p.shape))
                            pp += [p]
                    if not found:
                        p.requires_grad = False

            args.fout.write('tot_pars,' + str(totpars) + '\n')
            if (args.optimizer_type=='Adam'):
                sel.temp.optimizer = optim.Adam(pp, weight_decay=.00002)
            else:
                sel.temp.optimizer = optim.SGD(pp,lr=args.lr)


class STVAE_mix(nn.Module):


    def __init__(self, sh, device, args, opt_setup=True):
        super(STVAE_mix, self).__init__()

        self.temp=temp_args()
        self.dv = device
        self.binary_thresh=args.binary_thresh
        self.initial_shape=sh
        self.opt = args.OPT
        self.opt_jump=args.opt_jump
        self.mu_lr = args.mu_lr
        self.lr=args.lr
        self.perturb=args.perturb
        self.s_dim = args.sdim
        self.n_mix = args.n_mix
        self.lower=args.lower_decoder
        self.nosep=args.nosep
        self.nti=args.nti
        self.n_class=args.n_class
        self.CC = args.CC
        self.optimizer_type = args.optimizer_type
        self.flow_prior=args.flow_prior
        if args.output_cont>0.:
            self.output_cont=args.output_cont #nn.Parameter(torch.tensor(args.output_cont), requires_grad=False)
        else:
            self.output_cont=0.
        print('OUTPUT CONT',self.output_cont)
        self.u_dim = 0
        setup_trans_stuff(self,args,sh, device)
        self.encoder_m = encoder_mix(sh, device, args)
        self.final_shape=self.encoder_m.final_shape
        if args.OPT:
            self.encoder_m=None
            temp_layers_dict = get_network(args.dec_layers_top)
            self.final_shape=[l['num_units'] for l in temp_layers_dict if 'dense_gaus' in l['name']]

        self.dec_trans_top=None
        trans_shape=None
        dec_shape=copy.copy(self.final_shape)
        if self.u_dim>0:
            trans_shape=copy.copy(dec_shape)
            trans_shape[0]=self.u_dim
            dec_shape[0]=self.z_dim

        self.decoder_m = decoder_mix(self.u_dim, self.z_dim,trans_shape,dec_shape,device,args)
        if self.flow_prior:
            self.flow=flow_mix(self.z_dim, dec_shape,device,args)
        #,requires_grad=False)
        if self.n_class>1:
            self.rho=nn.Parameter(torch.zeros(self.n_class,self.n_mix//self.n_class))
        else:
            self.rho = nn.Parameter(torch.zeros(self.n_mix))
        self.scheduler=None
        if (not args.nosep and opt_setup):
            setup_optimizer(self,args)


    def encoder_mix(self,input):

        return self.encoder_m(input)

    def decoder_mix(self, input, rng=None, lower=False):

       if not lower:
            return self.decoder_m(input, rng)
       else:
           return self.decoder_m.lower_forward(input)




    def update_s(self,mu,logvar,pi,mu_lr, prop=None, both=True):

        var={}

        var['mu']=torch.autograd.Variable(mu.to(self.dv), requires_grad=True)
        var['logvar'] = torch.autograd.Variable(logvar.to(self.dv), requires_grad=True)
        var['pi'] = torch.autograd.Variable(pi.to(self.dv), requires_grad=True)
        if prop is not None:
            var['prop']=torch.autograd.Variable(prop.to(self.dv), requires_grad=True)

        PP1=[vals for keys,vals in var.items()]
        PP2 = []
        if both:
            for p in self.parameters():
                PP2+=[p]

        if self.optimizer_type=='Adam':
            self.optimizer_s = optim.Adam([{'params':PP1,'lr':mu_lr},
                                      {'params':PP2}],lr=self.lr)
        else:
            self.optimizer_s = optim.SGD([{'params': PP1, 'lr': mu_lr},
                                           {'params': PP2}], lr=self.lr)

        return var



    def decoder_and_trans(self,s, rng=None, train=True):

        n_mix=s.shape[0]

        x, u = self.decoder_mix(s, rng=rng, lower=self.lower)
        # Transform
        if (self.u_dim>0):
            xt = []
            for xx,uu in zip(x,u):
                    xt=xt+[self.apply_trans(xx,uu).squeeze()]

            x=torch.stack(xt,dim=0).reshape(n_mix,x.shape[1],-1)
        xx = torch.clamp(x, self.binary_thresh, 1 - self.binary_thresh)
        return xx


    def sample(self, var, dim):
        #print(mu.shape,dim)
        eps=torch.randn(var['mu'].shape).to(self.dv)

        #eps = torch.randn(mu.shape[0],dim).to(self.dv)
        if (self.s_dim>1):
            z = var['mu'] + torch.exp(var['logvar']/2) * eps
        else:
            z = torch.ones(var['mu'].shape[0],dim).to(self.dv)
        return z

    def mixed_loss_pre(self,x,data):
        b = []
        ninp=data.shape[0]

        if (self.output_cont==0.):
            for xx in x:
                a = F.binary_cross_entropy(xx.reshape(data.shape[0],-1), data.reshape(data.shape[0], -1),
                                           reduction='none')
                a = torch.sum(a, dim=1)
                b = b + [a]
        else:
            for xx in x:
                #datas=data.reshape(data.shape[0],-1)
                a=(data.reshape(ninp,-1)-xx.reshape(ninp,-1))*(data.reshape(ninp,-1)-xx.reshape(ninp,-1))
                a = torch.sum(a, dim=1)*self.output_cont
                b = b + [a]
        b = torch.stack(b).transpose(0, 1)
        return(b)

    def weighted_sum_of_likelihoods(self,lpi,b):
        return(-torch.logsumexp(lpi-b,dim=1))

    def mixed_loss_inter(self,x,data,pi):
        b = self.mixed_loss_pre(x, data)
        recloss = torch.sum(pi * b,axis=1)
        return recloss
    def mixed_loss(self,x,data,pi):

        b=self.mixed_loss_pre(x,data)
        recloss=torch.sum(pi*b)
        return recloss


    def get_loss(self,data_to_match,targ,var,rng=None, back_ground=None):

        if back_ground is not None:
            return self.get_loss_background(data_to_match,var,back_ground,rng=rng)

        pi=var['pi']
        mu= var['mu'] if 'mu' in var else torch.cat((var['muu'],var['mus']),dim=1)
        logvar=var['logvar']

        if (targ is not None):
            pi=pi.reshape(-1,self.n_class,self.n_mix_perclass)
            pi=torch.softmax(pi,dim=2)
            #pis=torch.sum(pi,2)
            #pi = pi/pis.unsqueeze(2)
        else:
            pi=torch.softmax(pi, dim=1)
        lpi = torch.log(pi)
        n_mix = self.n_mix
        if (targ is None and self.n_class > 1):
            n_mix = self.n_mix_perclass
        if (self.type != 'ae'):
            s = self.sample(var, mu.shape[1])
        else:
            s = mu
        s = s.reshape([s.shape[0], n_mix, s.shape[1] // n_mix] + list(s.shape[2:])).transpose(0, 1)
        # Apply linear map to entire sampled vector.
        x = self.decoder_and_trans(s, rng)
        if back_ground is not None:
            xx = []
            for xi, bi in zip(x, back_ground):
                xi = xi * (xi >= .5) + bi * (xi < .5)
                xx += [xi]
            x = torch.stack(xx, dim=0)
        if (targ is not None):
            x = x.transpose(0, 1)
            x = x.reshape([x.shape[0], self.n_class, self.n_mix_perclass]+list(x.shape[2:]))
            mu = mu.reshape(mu.shape[0], self.n_class, -1)
            logvar = logvar.reshape(logvar.shape[0], self.n_class, -1)

            tot = 0
            recloss = 0
            if (type(targ) == torch.Tensor):
                for c in range(self.n_class):
                    ind = (targ == c)
                    if self.type != 'ae':
                        tot += dens_apply(self.rho[c], mu[ind, c, :], logvar[ind, c, :],  lpi[ind, c, :], pi[ind, c, :])[0]
                    recloss += self.mixed_loss(x[ind, c, :, :].transpose(0, 1), data_to_match[ind], pi[ind, c, :])
            else:
                if self.type != 'ae':
                    tot += dens_apply(self.rho, mu[:, targ, :], logvar[:, targ, :],lpi[:, targ, :], pi[:, targ, :])[0]
                recloss += self.mixed_loss(x[:, targ, :, :].transpose(0, 1), data_to_match, pi[:, targ, :])
        else:
            tot = 0.
            if (self.type != 'ae'):
                tot, _, _ = dens_apply(self.rho, mu, logvar,  lpi, pi)
            recloss = self.mixed_loss(x, data_to_match, pi)
        return recloss, tot, x, None


    def get_loss_background(self,data_to_match, var, backs, rng=None):

        pi=var['pi'];
        s= var['mu'] if 'mu' in var else torch.cat((var['muu'],var['mus']),dim=1)

        pi=torch.softmax(pi,dim=1)
        n_mix = pi.shape[1]
        num=s.shape[0]
        #in_feats=self.decoder_m.in_feats
        #in_shape=self.decoder_m.in_shape
        s=s.reshape([num,n_mix,s.shape[1]//n_mix]+list(s.shape[2:])).transpose(0,1)
        #objs=self.decoder_m.top_forward(s)[0]
        x = self.decoder_m.forward(s,rng=rng)[0]
        pmix=torch.sigmoid(var['prop'])
        #back=backs['mu'].reshape(1,-1,in_feats,in_shape[0],in_shape[1])
        #xb=self.decoder_m.lower_forward(back)[0]
        sm=[]
        xm=[]
        for xx in x:
            #xi = xi * (xi >= .5) + bi * (xi < .5)
            #x=self.decoder_m.forward(s)[0]
            #xm+=[pmix*x+(1-pmix)*xb]
            xx=xx.reshape((-1,)+self.initial_shape)
            xm+=[pmix*xx+(1-pmix)*data_to_match]
            #sm+= [obs.reshape(-1,in_feats,in_shape[0],in_shape[1])*(pmix>=.5)+back*(pmix<.5)]
            #sm+=[pmix*obs.reshape(-1,in_feats,in_shape[0],in_shape[1])+(1-pmix)*back]
        #sm=torch.stack(sm,dim=0)
        xm=torch.stack(xm,dim=0)

        #x=self.decoder_m.lower_forward(sm)[0]
        #prior_s=torch.sum(s*s)
        prior_s=0
        prior_b=-torch.sum(pmix)*self.CC #torch.sum((1-pmix)*(back*back)*self.CC)#-torch.sum((1-pmix)*torch.log(1-pmix))
        recloss = self.mixed_loss(xm, data_to_match, pi)
        tot=prior_s+prior_b
        #print('prior_b',prior_b/num)
        return recloss, tot, xm, pmix



    def encoder_and_loss(self,var, data, data_to_match, targ, rng,back_ground=None):

        #with torch.no_grad() if not self.flag else dummy_context_mgr():

        # if (self.opt):
        #        # var['spi'] = torch.softmax(var['pi'], dim=1)
        #        pass
        # else:
        #         var, _ = self.encoder_mix(data)
        tot=0
        x=None
        bla=None
        #recloss, tot, x, bla =self.get_loss(data_to_match,targ, var, rng,  back_ground=back_ground)
        #recloss+=tot
        recloss, tot = self.compute_likelihood_for_batch(data_to_match,1)
        recloss=-recloss
        tot=-tot
        return recloss, tot, x, bla


    def compute_loss_and_grad(self, var, data,data_to_match,targ,d_type,optim, opt='par', rng=None,back_ground=None):

        optim.zero_grad()

        recloss, tot, x, pmix = self.encoder_and_loss(var, data,data_to_match,targ,rng,back_ground=back_ground)
        loss = recloss

        if (d_type == 'train' or opt=='mu'):

            loss.backward(retain_graph=(opt=='mu'))

            optim.step()

        return recloss.item(), tot.item(), x, pmix



    def get_logdets(self):
            return

    def run_epoch(self, args, train, epoch,num_mu_iter, mu, logvar, pi, d_type='test',fout=None, cl=None):


        if (d_type=='train'):
            self.train()
        else:
            self.eval()

        tr_loss = 0;tr_kl = 0
        self.epoch=epoch

        tra=iter(train)
        for j in np.arange(0, train.num, train.batch_size):
            BB, indlist=next(tra)
            data_in=BB[0].to(self.dv)
            data=BB[0].to(self.dv)
            if self.perturb > 0. and d_type == 'train' and not self.opt:
                with torch.no_grad():
                    data = deform_data(data, self.perturb, self.transformation, self.s_factor, self.h_factor, True, self.dv)
            #if self.perturb<0 and d_type=='train':
            #    data=add_clutter(data)
            data_d = data.detach()
            target=None
            if (self.n_class>1):
                target = BB[1].to(self.dv)
            if self.opt:
                var=self.update_s(mu[indlist, :], logvar[indlist, :], pi[indlist], self.mu_lr[0],both=self.nosep)
                if np.mod(epoch, self.opt_jump) == 0:
                  for it in range(num_mu_iter):
                    loss,kl, _, _  = self.compute_loss_and_grad(var, data_in, data_d, target, d_type, self.optimizer_s, opt='mu')
                ppi=torch.softmax(var['pi'],axis=1)
                cll=torch.argmax(ppi,axis=1)
                ccc=torch.unique(cll,return_counts=True)
                print('h',ccc)
            else:
                var={}
            if not self.opt or not self.nosep:
              with torch.no_grad() if (d_type != 'train') else dummy_context_mgr():
                    loss, kl, _, _ =self.compute_loss_and_grad(var, data_in, data, target,d_type,self.temp.optimizer)

            if self.opt:
                mu[indlist] = var['mu'].data.cpu()
                logvar[indlist] = var['logvar'].data.cpu()
                pi[indlist] = var['pi'].data.cpu()

            tr_loss += loss
            tr_kl += kl
            if d_type=='test':
                for i,b in enumerate(BB[1]):
                    print("CLUST:",b.data.cpu().numpy(),np.argmax(pi[indlist[i]].numpy()))

        if (True): #(np.mod(epoch, 10) == 9 or epoch == 0):
            fout.write('\n====> Epoch {}: {} loss: {:.4f}, kl: {:.4F}\n'.format(d_type,
        epoch, tr_loss / train.num, tr_kl/train.num))
            fout.write('RHO:')
            for r in torch.softmax(self.rho,0).detach().cpu().numpy():
                fout.write('{:.3f} '.format(r))
        return mu, logvar, pi, [tr_loss/train.num, tr_kl / train.num]

    def recon(self,args, input,num_mu_iter=None, lower=False,back_ground=None):

            enc_out=None
            self.eval()
            if not lower:
                self.lower=False
                shape=self.final_shape
            else:
                self.lower=True
                self.opt=True
                shape=self.decoder_m.dec_conv_bot.temp.input_shape

            if self.opt:
                mu, logvar, ppi = initialize_mus(input.shape[0], shape, self.n_mix)

            num_inp=input.shape[0]
            #self.setup_id(num_inp)
            inp = input.float().to(self.dv)
            inp_d=inp.detach()


            if self.opt:
                prop=None
                lrr=self.mu_lr[0]
                if back_ground is not None:
                    lrr=self.mu_lr[1]
                    #dim1=self.decoder_m.in_feats
                    #prop=-2.*torch.ones([num_inp,dim1]+list(self.decoder_m.in_shape))
                    prop=-2.*torch.ones([num_inp]+list(self.initial_shape))
                    if type(back_ground) is dict: # This is for when background is the latent variables of the bot decoder.
                     for keys,values in back_ground.items():
                        back_ground[keys]=back_ground[keys].detach()
                var=self.update_s(mu, logvar, ppi, lrr,prop=prop, both=self.nosep)

                #torch.autograd.set_detect_anomaly(True)
                for it in range(num_mu_iter):
                    rls, ls, _, pmix = self.compute_loss_and_grad(var, inp_d,inp, None, 'test', self.optimizer_s, opt='mu', back_ground=back_ground)


            else:

                var, enc_out = self.encoder_mix(inp)

            #s = self.sample(s_mu, s_var, self.s_dim * self.n_mix)
                # for it in range(num_mu_iter):
                #     self.compute_loss_and_grad_mu(inp_d, s_mu, s_var, None, 'test', self.optimizer_s,
                #                                   opt='mu')
            var['pi'] = torch.softmax(var['pi'], dim=1)
            sh=[var['mu'].shape[0], self.n_mix, var['mu'].shape[1] // self.n_mix] + list(var['mu'].shape[2:])
            ss_mu = var['mu'].reshape(sh).transpose(0,1)

            ii = torch.argmax(var['pi'], dim=1)
            jj = torch.arange(0,num_inp,dtype=torch.int64).to(self.dv)
            kk = ii+jj*self.n_mix
            lpi = torch.log(var['pi'])
            totloss=0
            recon_batch_both=None
            with torch.no_grad():
                # if back_ground is not None:
                #    recon_batch = recon_batch * (recon_batch >= .2) + back_ground * (recon_batch < .2)
                if back_ground is not None and self.opt:
                    var['spi'] = var['pi']
                    recloss,totloss,recon_batch_both, pmix= self.get_loss_background(inp_d, var, back_ground)

                recon_batch = self.decoder_and_trans(ss_mu)
                #if pmix is not None:
                #    recon_batch *= (pmix>.5)
                recloss = self.mixed_loss_inter(recon_batch, inp, var['pi'])
                if back_ground is None and self.type != 'ae':
                    totloss = dens_apply(self.rho,var['mu'], var['logvar'],  lpi, var['pi'])
                #else:
                #    totloss=torch.sum(ss_mu*ss_mu)
            if  'prop' in var:
                print(torch.sigmoid(var['prop']))
            recon = recon_batch.transpose(0, 1)
            recon=recon.reshape(self.n_mix*num_inp,-1)

            rr=recon[kk]
            print('recloss',torch.sum(recloss)/num_inp,'totloss',totloss[0]/num_inp)
            return rr, var,[recloss,totloss[1]], enc_out, recon_batch, recon_batch_both



    def sample_from_z_prior(self,args, bsz, theta=None, clust=None, lower=False):


        self.eval()
        if not lower:
            self.lower = False
            shape = self.final_shape
        else:
            self.lower = True
            shape = self.decoder_m.dec_conv_bot.temp.input_shape

        rho_dist=torch.exp(self.rho-torch.logsumexp(self.rho,dim=0))
        if (clust is not None):
            ii=clust*torch.ones(bsz, dtype=torch.int64).to(self.dv)
        else:
            ii=torch.multinomial(rho_dist,bsz,replacement=True)
        sh=[bsz,args.n_mix]+list(shape)
        s = torch.randn(sh).to(self.dv)*self.CC

        if (theta is not None and self.u_dim>0):
            theta = theta.to(self.dv)
            for i in range(self.n_mix):
                s[:,i,0:self.u_dim]=theta
        s = s.transpose(0,1)
        if self.flow_prior:
            s=self.flow(s)
        with torch.no_grad():
            x=self.decoder_and_trans(s, train=False)

        x=x.transpose(0,1)
        jj = torch.arange(0, bsz, dtype=torch.int64).to(self.dv)
        kk = ii + jj * self.n_mix
        recon = x.reshape(self.n_mix * bsz, -1)
        rr = recon[kk]

        return rr



    def compute_likelihood_for_dataset(self,Input,num_samples):

       self.eval()

       num_inp=Input.num
       print('num_inp',num_inp)
       bsz=int(Input.batch_size/(2*num_samples))
       print(bsz)
       #LGS=[]
       LGG=0
       #EEE = torch.randn(num_samples, num_inp, self.n_mix, self.s_dim).to(self.dv)

       tra=iter(Input)
       tra=iter(Input)
       for j in np.arange(0, num_inp, Input.batch_size):
           input = next(tra)
           input = input[0][0]

           for jj in range(0,int(input.shape[0]),bsz):
                loc_input=input[jj:jj+bsz].to(self.dv)
                print(loc_input.shape)
                LG, _ =self.compute_likelihood_for_batch(loc_input,num_samples)

                LGG -= LG.detach().cpu().numpy() / num_inp
                loc_input.detach()


       print('LLL',LGG)

       return(LGG)


    def compute_likelihood_for_batch(self, input, num_samples):


        lns = np.log(num_samples)
        bsz = input.shape[0]
        EE = torch.randn(num_samples, bsz, self.n_mix, self.s_dim).to(self.dv)

        if self.opt:
            mu, logvar, ppi = self.initialize_mus(bsz, True)
            inp_d = input.detach()
            var = self.update_s(mu, logvar, ppi, self.mu_lr[0], both=self.nosep)
            # self.get_logdets()
            for it in range(self.nti):
                self.compute_loss_and_grad(var, inp_d, input, None, 'test', self.optimizer_s, opt='mu')
        else:
            var, _ = self.encoder_mix(input)

        s_var = var['logvar'].reshape(1, bsz, self.n_mix, self.s_dim)
        tsd = torch.exp(s_var / 2)
        tmu = var['mu'].reshape(1, bsz, self.n_mix, self.s_dim)
        S = tmu + tsd * EE

        #with torch.no_grad():

        recon_batch = self.decoder_and_trans(S.reshape(-1, self.n_mix, self.s_dim).transpose(0, 1))
        inp = input.repeat(num_samples, 1, 1, 1)
        recon_loss = -self.mixed_loss_pre(recon_batch, inp).reshape(num_samples, bsz, self.n_mix)

        logq = -torch.sum((S - tmu) * (S - tmu) / (2 * tsd * tsd), dim=3) \
               - torch.sum(s_var, dim=3) / 2  # + torch.log(pi.unsqueeze(0))

        if self.flow_prior:
            S, logp = self.flow.inv_and_logdet(S.reshape(-1, self.n_mix, self.s_dim).transpose(0, 1))
            S=S.reshape(self.n_mix,num_samples,bsz,self.s_dim)
            logp=logp.reshape(self.n_mix,num_samples,bsz)
            logp -= .5 * torch.sum(S * S, dim=3)
            logp = logp.permute(1,2,0)
        else:
            logp = -.5 * torch.sum(S * S, dim=3)

        pim = torch.softmax(var['pi'], 1)  # .repeat(num_samples, 1), 1)
        lpim = torch.log_softmax(var['pi'].repeat(num_samples, 1), 1).reshape(num_samples, bsz, self.n_mix)
        lsfrho = torch.log_softmax(self.rho, 0).repeat(num_samples * bsz, 1).reshape(num_samples, bsz, self.n_mix)
        LGp=None
        LGP=0
        if num_samples==1:
            LGr = recon_loss
            LGp = logp - logq + lsfrho - lpim
            LG=LGr+LGp
            LGP=torch.sum(LGp)
        else:
            LG = torch.logsumexp(recon_loss + logp - logq + lsfrho - lpim, 0)
        LG = torch.sum(LG * pim, 1)
        LG -= lns
        # LGS += [LG]
        LG = torch.sum(LG)
        return LG, LGP

    def get_scheduler(self,args):
        self.scheduler=None
        if args.sched==1.:
            l2 = lambda epoch: pow((1.-1. * epoch/args.nepoch),0.9)
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.temp.optimizer, lr_lambda=l2)
        elif args.sched==2.:
            self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.temp.optimizer,verbose=True)
        elif args.sched==3:
            self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.temp.optimizer,milestones=[10,20], gamma=.5)


