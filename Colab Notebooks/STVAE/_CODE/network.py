import numpy as np
import torch
import time
import torch.nn.functional as F
from torch import nn, optim
import contextlib
from models_transforms import Edge, rgb_to_hsv, hsv_to_rgb
import sys
from model_layers import FALinear, FAConv2d
from models_residual_block import residual_block, residual_block_small
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

class hinge_loss(nn.Module):
    def __init__(self, mu=1., num_class=10):
        super(hinge_loss, self).__init__()
        self.fac = mu/(num_class-1)
        self.ee = torch.eye(num_class)

    def forward(self, input, target):

        targarr = self.ee[target] > 0
        loss = torch.sum(torch.relu(1 - input[targarr])) + self.fac * torch.sum(
            torch.relu(1 + input[torch.logical_not(targarr)]))
        loss /= input.shape[0]
        return loss

class Reshape(nn.Module):
    def __init__(self,sh):
        super(Reshape,self).__init__()

        self.sh=sh

    def forward(self, input):

        out=torch.reshape(input,[-1]+self.sh)

        return(out)

# Network module
class network(nn.Module):
    def __init__(self, device,  args, layers, lnti, fout=None, sh=None, first=1):
        super(network, self).__init__()

        self.trans=args.transformation
        self.wd=args.wd
        self.embedd=args.embedd
        self.embedd_layer=args.embedd_layer
        self.del_last=args.del_last
        self.first=first
        self.bsz=args.mb_size # Batch size - gets multiplied by number of shifts so needs to be quite small.
        #self.full_dim=args.full_dim
        self.dv=device
        self.edges=args.edges
        self.update_layers=args.update_layers
        self.n_class=args.n_class
        self.s_factor=args.s_factor
        self.h_factor=args.h_factor
        #self.pools = args.pools # List of pooling at each level of network
        #self.drops=args.drops # Drop fraction at each level of network
        self.optimizer_type=args.optimizer
        self.lr=args.lr
        self.layer_text=layers
        self.fa=args.fa
        self.lnti=lnti
        self.HT=nn.Hardtanh(0.,1.)
        self.back=('vae' in args.type)
        if fout is not None:
            self.fout=fout
        else:
            self.fout=sys.stdout
        self.embedd_type=args.embedd_type
        self.ed = Edge(self.dv, dtr=.03).to(self.dv)
        # The loss function
        if args.hinge:
            self.criterion=hinge_loss(num_class=args.num_class)
        else:
            self.criterion=nn.CrossEntropyLoss()
        #self.crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(500.))

        if (hasattr(args,'perturb')):
            self.perturb=args.perturb
        self.u_dim = 6
        self.idty = torch.cat((torch.eye(2), torch.zeros(2).unsqueeze(1)), dim=1)
        self.id = self.idty.expand((self.bsz,) + self.idty.size()) #.to(self.dv)

        if sh is not None and self.first:
            temp = torch.zeros([1]+list(sh[1:])) #.to(device)
            # Run the network once on dummy data to get the correct dimensions.
            bb = self.forward(temp)
            #self.get_seq(self.dv)
            self.output_shape=bb[0].shape
        self.get_scheduler(args)
    def do_nonlinearity(self,ll,out):

        if ('non_linearity' not in ll):
            return(out)
        elif ('HardT' in ll['non_linearity']):
            return(self.HT(out))
        elif ('tanh' in ll['non_linearity']):
            return(F.tanh(out))
        elif ('relu' in ll['non_linearity']):
            return(F.relu(out))



    def forward(self,input,everything=False):

        out = input
        in_dims=[]
        if (self.first):
            self.layers = nn.ModuleList()
            if self.back:
                self.back_layers=nn.ModuleList()
        OUTS={}
        old_name=''
        layer_text_new={}
        for i,ll in enumerate(self.layer_text):
                inp_ind = old_name

                if ('parent' in ll):
                    pp=ll['parent']
                    # over ride default inp_feats
                    if len(pp)==1:
                        inp_ind=pp[0] #self.lnti[pp[0]]
                        if self.first:
                            inp_feats=OUTS[pp[0]].shape[1] #self.layer_text[self.lnti[pp[0]]]['num_filters']
                            in_dim=in_dims[self.lnti[pp[0]]]
                    else:
                        inp_feats=[]
                        loc_in_dims=[]
                        inp_ind=[]
                        for p in pp:
                            inp_ind += p #[self.lnti[p]]
                            if self.first:
                                inp_feats+=[OUTS[p].shape[1]]
                                loc_in_dims+=[in_dims[self.lnti[p]]]
                if ('input' in ll['name']):
                    #if self.first or everything:
                    OUTS[ll['name']]=input
                    enc_hw=input.shape[2:4]

                if ('conv' in ll['name']):
                    if self.first:
                        bis = True
                        if ('nb' in ll):
                            bis = False
                        stride=1;
                        if 'stride' in ll:
                            stride=ll['stride']
                        #pd=tuple(np.int32(np.floor(np.array(ll['filter_size'])/2)))
                        pd=(ll['filter_size']//stride) // 2
                        self.layers.add_module(ll['name'],FAConv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,fa=self.fa,padding=pd, bias=bis, device=self.dv))
                        #self.layers.add_module(ll['name'],nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=1,padding=pd))
                        if self.back:
                            self.back_layers.add_module(ll['name']+'_bk',nn.Conv2d(ll['num_filters'],inp_feats,ll['filter_size'],stride=stride,padding=pd))
                    out=self.do_nonlinearity(ll,getattr(self.layers, ll['name'])(OUTS[inp_ind]))
                    #if self.first or everything:
                    OUTS[ll['name']]=out
                if ('Avg' in ll['name']):
                    if self.first:
                        HW=(np.int32(OUTS[inp_ind].shape[2]/2),np.int32(OUTS[inp_ind].shape[3]/2))
                        self.layers.add_module(ll['name'],nn.AvgPool2d(HW,HW))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']] = out
                if ('pool' in ll['name']):
                    if self.first:
                        stride = ll['pool_size']
                        if ('stride' in ll):
                            stride = ll['stride']
                        pp=[np.int32(np.mod(ll['pool_size'],2))]
                        self.layers.add_module(ll['name'],nn.MaxPool2d(ll['pool_size'], stride=stride, padding=pp))
                        if self.back:
                            self.back_layers.add_module(ll['name']+'_bk',nn.UpsamplingNearest2d(scale_factor=stride))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out


                if ('drop' in ll['name']):
                    if self.first:
                        self.layers.add_module(ll['name'],torch.nn.Dropout(p=ll['drop'], inplace=False))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    #if self.first or everything:
                    OUTS[ll['name']]=out

                if ('dense' in ll['name']):
                    if self.first:
                        out_dim=ll['num_units']
                        bis=True
                        if ('nb' in ll):
                            bis=False
                        self.layers.add_module(ll['name'],FALinear(in_dim,out_dim,bias=bis, fa=self.fa))
                        #self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis).to(self.dv))

                        if self.back:
                            self.back_layers.add_module(ll['name']+'_bk_'+'reshape',Reshape(list(OUTS[inp_ind].shape[1:])))
                            self.back_layers.add_module(ll['name']+'_bk',nn.Linear(out_dim,in_dim,bias=bis))
                    out=OUTS[inp_ind]
                    out = out.reshape(out.shape[0], -1)
                    out=self.do_nonlinearity(ll,getattr(self.layers, ll['name'])(out))
                    #if self.first or everything:
                    OUTS[ll['name']]=out

                if ('norm') in ll['name']:
                    if self.first:
                        if len(OUTS[old_name].shape)==4:
                            self.layers.add_module(ll['name'],torch.nn.BatchNorm2d(OUTS[old_name].shape[1]))
                        else:
                            self.layers.add_module(ll['name'],torch.nn.BatchNorm1d(OUTS[old_name].shape[1]))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out

                if ('res' in ll['name']):
                    if self.first:
                        pd=tuple(np.int32(np.floor(np.array(ll['filter_size'])/2)))
                        self.layers.add_module(ll['name'],residual_block(inp_feats,ll['num_filters'],self.dv,stride=1,pd=pd))
                    out=self.do_nonlinearity(ll,getattr(self.layers, ll['name'])(OUTS[inp_ind]))
                    OUTS[ll['name']]=out

                if ('opr' in ll['name']):
                    if 'add' in ll['name'] and everything:
                        out = OUTS[inp_ind[0]]+OUTS[inp_ind[1]]
                        OUTS[ll['name']] = out
                        inp_feats=out.shape[1]
                if ('num_filters' in ll):
                    inp_feats = ll['num_filters']
                if self.first==1:
                    self.fout.write(ll['name']+' '+str(np.array(OUTS[ll['name']].shape))+'\n')
                in_dim=np.prod(OUTS[ll['name']].shape[1:])
                in_dims+=[in_dim]
                old_name=ll['name']


        if self.first==1:
            print(self.layers)
            tot_pars = 0
            KEYS=[]
            for keys, vals in self.state_dict().items():
                if self.first==1:
                    self.fout.write(keys + ',' + str(np.array(vals.shape))+'\n')
                if 'running' not in keys and 'tracked' not in keys:
                    KEYS+=[keys]
                tot_pars += np.prod(np.array(vals.shape))
            if self.first==1:
                self.fout.write('tot_pars,' + str(tot_pars)+'\n')
            # TEMPORARY
            pp=[]
            for k,p in zip(KEYS,self.parameters()):
                if (self.update_layers is None):
                    if self.first==1:
                        self.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                    pp+=[p]
                else:
                    found = False
                    for u in self.update_layers:
                        if u == k.split('.')[1]:
                            found=True
                            if self.first==1:
                                self.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                            pp+=[p]
                    if not found:
                        p.requires_grad=False

            if (self.optimizer_type == 'Adam'):
                if self.first==1:
                    self.fout.write('Optimizer Adam '+str(self.lr)+'\n')
                self.optimizer = optim.Adam(pp, lr=self.lr,weight_decay=self.wd)
            else:
                if self.first==1:
                    self.fout.write('Optimizer SGD '+str(self.lr))
                self.optimizer = optim.SGD(pp, lr=self.lr,weight_decay=self.wd)
        out1=[]
        if self.first:
            self.first=0
        if(everything):
            out1=OUTS
        #elif (len(OUTS) > 3):
         #   out1 = OUTS[-3]
        return(out,out1)

    def backwards(self,x):
        xx=x
        for l in reversed(list(self.back_layers)):
            xx=l(xx)

        return xx

        # Get loss and accuracy (all characters and non-space characters).
    def get_acc_and_loss(self, out, targ):
            v, mx = torch.max(out, 1)
            # Non-space characters
            # Total loss
            loss = self.criterion(out, targ)
            # total accuracy
            acc = torch.sum(mx.eq(targ))
            return loss, acc

    def standardize(self,out):

        outa=out.reshape(out.shape[0],-1)#-torch.mean(out,dim=1).reshape(-1,1)
        #out_a = torch.sign(outa) / out.shape[1]
        sd = torch.sqrt(torch.sum(outa * outa, dim=1)).reshape(-1, 1)
        out_a = outa/(sd+.01)

        return out_a

    def get_embedd_loss_new(self,out0,out1):
        thr = 2.
        #out0=torch.tanh(out0) 
        out0=self.standardize(out0)
        #out1=torch.tanh(out1) 
        out1=self.standardize(out1)
        out0b = out0.repeat([self.bsz, 1])
        out1b = out1.repeat_interleave(self.bsz, dim=0)
        outd = out0b - out1b
        outd = torch.sum(torch.relu(outd) + torch.relu(-outd), dim=1)
        OUT = -outd.reshape(self.bsz, self.bsz).transpose(0, 1)
        # Multiply by y=-1/1
        OUT=(OUT+thr)*(2.*torch.eye(self.bsz).to(self.dv)-1.)

        loss=torch.sum(torch.relu(1-OUT))


        acc=torch.sum(OUT>0).type(torch.float)/self.bsz

        return loss,acc

    def get_embedd_loss_new_a(self, out0, out1):

        thr1=.9
        thr2=.3
        thr=(thr1+thr2)*.5
        # Standardize 64 dim outputs of original and deformed images
        out0a = self.standardize(out0)
        out1a = self.standardize(out1)
        # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
        #COV = torch.mm(out0a, out1a.transpose(0, 1))
        COV = torch.mm(out0a, out1a.transpose(0, 1))
        cc = COV.flatten()
        targ = torch.zeros(cc.shape[0],dtype=torch.bool).to(self.dv)
        targ[0:cc.shape[0]:(COV.shape[0] + 1)] = 1
        cc1=torch.relu(thr1-cc[targ])
        cc2=torch.relu(cc[targ.logical_not()]-thr2)
        loss1=torch.sum(cc1)
        loss2=torch.sum(cc2)
        loss=(loss1+loss2)/(self.bsz*self.bsz)

        acc = (torch.sum(cc[targ]>thr)+torch.sum(cc[targ.logical_not()]<thr)).type(torch.float) / self.bsz

        return loss, acc


    def get_embedd_loss_binary(self, out0, out1):

        # Standardize 64 dim outputs of original and deformed images
        out0a = self.standardize(out0)
        out1a = self.standardize(out1)
        # Compute 3 covariance matrices - 0-1, 0-0, 1-1.
        COV = torch.mm(out0a, out1a.transpose(0, 1))
        cc=COV.flatten()
        targ=torch.zeros(cc.shape[0]).to(self.dv)
        targ[0:cc.shape[0]:(COV.shape[0]+1)]=1
        loss=F.binary_cross_entropy_with_logits(cc,targ,pos_weight=torch.tensor(float(self.bsz)))


        icov = (cc-.75) * (2.*targ-1.)
        acc = torch.sum((icov > 0).type(torch.float)) / self.bsz

        return loss, acc

    def get_embedd_loss(self,out0,out1):

        out0a=self.standardize(out0)
        out1a=self.standardize(out1)
        COV=torch.mm(out0a,out1a.transpose(0,1))
        COV1 = torch.mm(out1a, out1a.transpose(0, 1))
        COV0 = torch.mm(out0a,out0a.transpose(0,1))
        vb=(torch.eye(self.bsz)*1e10).to(self.dv)

        cc = torch.cat((COV, COV0 - vb), dim=1)
        targ = torch.arange(self.bsz).to(self.dv)
        l1 = self.criterion(cc, targ)
        cc = torch.cat((COV.T, COV1 - vb), dim=1)
        l2 = self.criterion(cc, targ)
        loss =  (l1 + l2) / 2

        ID=2.*torch.eye(out0.shape[0]).to(self.dv)-1.
        icov=ID*COV

        acc=torch.sum((icov>0).type(torch.float))/self.bsz
        return loss,acc


    # GRADIENT STEP
    def loss_and_acc(self, input, target):
        #t0 = time.time()


        # Get output of network
        if type(input) is list:
            out0,ot0=self.forward(input[0])
            out1,ot1=self.forward(input[1])
            if self.embedd_type=='orig':
                loss, acc = self.get_embedd_loss(out0,out1)
            elif self.embedd_type=='binary':
                loss, acc = self.get_embedd_loss_binary(out0,out1)
            elif self.embedd_type=='L1dist_hinge':
                loss, acc = self.get_embedd_loss_new(out0,out1)
            else:
                loss, acc = self.get_embedd_loss_new_a(out0, out1)
        else:
            out,_=self.forward(input)
            # Compute loss and accuracy
            loss, acc=self.get_acc_and_loss(out,target)

        # Perform optimizer step using loss as criterion
        #t1 = time.time()

        #print('bak+for',time.time() - t0)
        return loss, acc

    def deform_data_old(self,x_in):
        h=x_in.shape[2]
        w=x_in.shape[3]
        nn=x_in.shape[0]
        u=((torch.rand(nn,6)-.5)*self.perturb).to(self.dv)
    # Ammplify the shift part of the
        u[:,[2,5]]*=2
        # Just shift and sclae
        #u[:,0]=u[:,4]
        #u[:,[1,3]]=0
        rr = torch.zeros(nn, 6).to(self.dv)
        rr[:, 0] = 1
        rr[:, 4] = 1
        theta = (u+rr).view(-1, 2, 3) #+ self.id
        grid = F.affine_grid(theta, [nn,1,h,w],align_corners=True)
        x_out=F.grid_sample(x_in,grid,padding_mode='zeros',align_corners=True)

        if x_in.shape[1]==3:
            v=torch.rand(nn,2).to(self.dv)
            vv=torch.pow(2,(v[:,0]*self.s_factor-self.s_factor/2)).reshape(nn,1,1)
            uu=((v[:,1]-.5)*self.h_factor).reshape(nn,1,1)
            x_out_hsv=rgb_to_hsv(x_out,self.dv)
            x_out_hsv[:,1,:,:]=torch.clamp(x_out_hsv[:,1,:,:]*vv,0.,1.)
            x_out_hsv[:,0,:,:]=torch.remainder(x_out_hsv[:,0,:,:]+uu,1.)
            x_out=hsv_to_rgb(x_out_hsv,self.dv)
        return x_out


    def deform_data(self,x_in):
        h=x_in.shape[2]
        w=x_in.shape[3]
        nn=x_in.shape[0]
        u=((torch.rand(nn,6)-.5)*self.perturb).to(self.dv)
        # Ammplify the shift part of the
        u[:,[2,5]]*=2
        # Just shift and sclae
        #u[:,0]=u[:,4]
        #u[:,[1,3]]=0
        rr = torch.zeros(nn, 6).to(self.dv)
        rr[:, [0,4]] = 1
        if self.trans=='shift':
          u[:,[0,1,3,4]]=0
        elif self.trans=='scale':
          u[:,[1,3]]=0
           #+ self.id
        elif 'rotate' in self.trans:
          u[:,[0,1,3,4]]*=1.5
          ang=u[:,0]
          v=torch.zeros(nn,6).to(self.dv)
          v[:,0]=torch.cos(ang)
          v[:,1]=-torch.sin(ang)
          v[:,4]=torch.cos(ang)
          v[:,3]=torch.sin(ang)
          s=torch.ones(nn).to(self.dv)
          if 'scale' in self.trans:
            s = torch.exp(u[:, 1])
            #print(s)
            #print(ang*180/np.pi)
          u[:,[0,1,3,4]]=v[:,[0,1,3,4]]*s.reshape(-1,1).expand(nn,4)
          rr[:,[0,4]]=0
        theta = (u+rr).view(-1, 2, 3)
        grid = F.affine_grid(theta, [nn,1,h,w],align_corners=True)
        x_out=F.grid_sample(x_in,grid,padding_mode='zeros',align_corners=True)

        if x_in.shape[1]==3 and self.s_factor>0:
            v=torch.rand(nn,2).to(self.dv)
            vv=torch.pow(2,(v[:,0]*self.s_factor-self.s_factor/2)).reshape(nn,1,1)
            uu=((v[:,1]-.5)*self.h_factor).reshape(nn,1,1)
            x_out_hsv=rgb_to_hsv(x_out,self.dv)
            x_out_hsv[:,1,:,:]=torch.clamp(x_out_hsv[:,1,:,:]*vv,0.,1.)
            x_out_hsv[:,0,:,:]=torch.remainder(x_out_hsv[:,0,:,:]+uu,1.)
            x_out=hsv_to_rgb(x_out_hsv,self.dv)

        # ii=torch.where(torch.bernoulli(torch.ones(self.bsz)*.5)==1)
        # for i in ii:
        #     x_out[i]=x_out[i].flip(3)
        return x_out


    # Epoch of network training
    def run_epoch(self, train, epoch, num_mu_iter=None, trainMU=None, trainLOGVAR=None, trPI=None, d_type='train', fout='OUT'):

        if (d_type=='train'):
            self.train()
        else:
            self.eval()
        num_tr=train[0].shape[0]
        #ii = np.arange(0, num_tr, 1)
        #if (type=='train'):
        #  np.random.shuffle(ii)
        jump = self.bsz
        trin = train[0] #[ii]
        targ = train[2] #[ii]
        self.n_class = np.max(targ) + 1

        full_loss=0; full_acc=0;
        # Loop over batches.

        targ_in=targ
        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            if (d_type == 'train'):
                self.optimizer.zero_grad()
            if self.embedd:
                with torch.no_grad():
                    data_in=(torch.from_numpy(trin[j:j + jump]).float()).to(self.dv)
                    data_out1=self.deform_data(data_in)
                    data=[data_in,data_out1]
            else:
                data = torch.from_numpy(trin[j:j + jump]).to(self.dv,dtype=torch.float32)

            target = torch.from_numpy(targ_in[j:j + jump]).to(self.dv, dtype=torch.long)

            with torch.no_grad() if (d_type!='train') else dummy_context_mgr():
                loss, acc= self.loss_and_acc(data, target)
            if (d_type == 'train'):
                loss.backward()
                self.optimizer.step()

            full_loss += loss
            full_acc += acc

        if (True):
            full_loss=np.float32(full_loss.detach().cpu().numpy())
            full_acc=np.float32(full_acc.detach().cpu().numpy())

            fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.4F} \n'.format(d_type,epoch,
                    full_loss /(num_tr/jump), full_acc/(num_tr)))

        return trainMU, trainLOGVAR, trPI, [full_acc/(num_tr), full_loss/(num_tr/jump)]

    def get_embedding(self, train):

        lay=self.embedd_layer
        trin = train
        jump = self.bsz
        num_tr = train.shape[0]
        self.eval()
        OUT=[]
        for j in np.arange(0, num_tr, jump, dtype=np.int32):
            data = (torch.from_numpy(trin[j:j + jump]).float()).to(self.dv)

            with torch.no_grad():
                out=self.forward(data, everything=True)[1][lay].detach().cpu().numpy()
                OUT+=[out]

        OUTA=np.concatenate(OUT,axis=0)

        return OUTA

    def get_scheduler(self,args):
        self.scheduler = None
        if args.sched>0.:
            self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[50,100,150,200,250,300,350],args.sched)
            #l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), args.sched)
            #scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)





