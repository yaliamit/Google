
from torch import nn, optim
import contextlib
from images import deform_data, Edge
from losses import *
import sys
from layers import *
import time

try:
    import torch_xla.core.xla_model as xm
except:
    pass
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

pre=get_pre()

osu=os.uname()


# Network module
class network(nn.Module):
    def __init__(self, device,  args, layers, lnti, fout=None, sh=None, first=1):
        super(network, self).__init__()

        self.grad_clip=args.grad_clip
        self.bn=args.bn
        self.trans=args.transformation
        self.wd=args.wd
        self.embedd=args.embedd
        self.embedd_layer=args.embedd_layer # Layer to use for embedding
        self.first=first
        self.future=args.future
        self.penalize_activations=args.penalize_activations
        self.bsz=args.mb_size # Batch size - gets multiplied by number of shifts so needs to be quite small.
        #self.full_dim=args.full_dim
        self.dv=device
        self.edges=args.edges
        self.update_layers=args.update_layers
        self.n_class=args.n_class
        self.s_factor=args.s_factor
        self.h_factor=args.h_factor
        self.optimizer_type=args.optimizer
        self.lr=args.lr
        self.layer_text=layers
        self.fa=args.fa
        self.randomize=args.layerwise_randomize
        self.lnti=lnti
        self.no_standardize=args.no_standardize
        if hasattr(args,'thr'):
            self.thr=args.thr
            self.delta=args.delta
        self.back=('ae' in args.type)
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
        self.CLR=SimCLRLoss(self.bsz,self.dv)
        #self.crit=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(500.))
        self.perturb=None
        if (hasattr(args,'perturb')):
            self.perturb=args.perturb

        if sh is not None and self.first:
            temp = torch.zeros([1]+list(sh[1:])) #.to(device)
            # Run the network once on dummy data to get the correct dimensions.
            dv=self.dv
            self.dv=torch.device("cpu")
            bb = self.forward(temp)
            self.dv=dv
            self.output_shape=bb[0].shape

        if fout is not None:
            fout.flush()


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
        prev_shape=None
        for i,ll in enumerate(self.layer_text):
                inp_ind = old_name

                if ('parent' in ll):
                    pp=ll['parent']
                    # over ride default inp_feats
                    if len(pp)==1:
                        inp_ind=pp[0]
                        if self.first:
                            inp_feats=OUTS[pp[0]].shape[1]
                            in_dim=in_dims[self.lnti[pp[0]]]
                    else:
                        inp_feats=[]
                        loc_in_dims=[]
                        inp_ind=[]
                        for p in pp:
                            inp_ind += [p]
                            if self.first:
                                inp_feats+=[OUTS[p].shape[1]]
                                loc_in_dims+=[in_dims[self.lnti[p]]]
                if ('input' in ll['name']):
                    OUTS[ll['name']]=input
                    enc_hw=input.shape[2:4]
                if ('shift' in ll['name']):
                     if self.first:
                         self.layers.add_module(ll['name'],shifts(ll['shifts']))
                     out=getattr(self.layers,ll['name'])(OUTS[inp_ind],self.dv)
                     OUTS[ll['name']]=out
                if ('conv' in ll['name']):
                    if self.first:
                        bis = True
                        if ('nb' in ll):
                            bis = False
                        stride=1;
                        if 'stride' in ll:
                            stride=ll['stride']
                        pd=(ll['filter_size']//stride) // 2
                        if not self.back:
                            if 'fa' in ll['name'] and not 'ga' in pre and 'Darwin' not in os.uname():
                                self.layers.add_module(ll['name'],FAConv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,fa=self.fa,padding=pd, bias=bis, device=self.dv))
                            else:
                                self.layers.add_module(ll['name'],nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,padding=pd, bias=bis))
                        else:
                            self.layers.add_module(ll['name'],nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=1,padding=pd))
                        if self.back:
                            self.back_layers.add_module(ll['name']+'_bk',nn.Conv2d(ll['num_filters'],inp_feats,ll['filter_size'],stride=1,padding=pd))
                    out=getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out
                if 'non_linearity' in ll['name']:
                    if self.first:
                        low=-1.; high=1.
                        if 'lims' in ll:
                            low=ll['lims'][0]; high=ll['lims'][1]
                        self.layers.add_module(ll['name'],NONLIN(ll,low=low,high=high))
                        if self.back:
                            self.back_layers.add_module(ll['name']+'_bk',NONLIN(ll,low=low,high=high))

                    OUTS[ll['name']] = getattr(self.layers, ll['name'])(OUTS[inp_ind])
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
                        pp=(ll['pool_size']-1)//2
                        self.layers.add_module(ll['name'],nn.MaxPool2d(ll['pool_size'], stride=stride, padding=pp))
                        if self.back:
                            #self.back_layers.add_module(ll['name']+'_bk',nn.UpsamplingNearest2d(scale_factor=stride))
                            self.back_layers.add_module(ll['name']+'_bk',nn.UpsamplingNearest2d(size=out.shape[2:4]))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out


                if ('drop' in ll['name']):
                    if self.first:
                        self.layers.add_module(ll['name'],torch.nn.Dropout(p=ll['drop'], inplace=False))
                        if self.back:
                            self.back_layers.add_module(ll['name']+'_bk',torch.nn.Dropout(p=ll['drop'],inplace=False))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out

                if ('dense' in ll['name']):
                    if self.first:
                        out_dim=ll['num_units']
                        bis=True
                        if ('nb' in ll):
                            bis=False
                        if not self.back:
                            if 'fa' in ll['name']:
                                self.layers.add_module(ll['name'],FALinear(in_dim,out_dim,bias=bis, fa=self.fa))
                            else:
                                self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis))
                        else:
                            self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis))

                        if self.back:
                            self.back_layers.add_module(ll['name']+'_bk_'+'reshape',Reshape(list(OUTS[inp_ind].shape[1:])))
                            self.back_layers.add_module(ll['name']+'_bk',nn.Linear(out_dim,in_dim,bias=bis))
                    out=OUTS[inp_ind]
                    out = out.reshape(out.shape[0], -1)
                    out=getattr(self.layers, ll['name'])(out)
                    OUTS[ll['name']]=out
                if 'subsample' in ll['name']:
                    if self.first:
                        stride = None
                        if 'stride' in ll:
                            stride = ll['stride']
                        self.layers.add_module(ll['name'], Subsample(stride=stride))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind],self.dv)
                    OUTS[ll['name']] = out
                if ('norm') in ll['name']:
                    if self.first:
                        if self.bn=='full':
                            if len(OUTS[old_name].shape)==4 and self.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[old_name].shape[1]))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[old_name].shape[1]))
                        elif self.bn=='half_full':
                            if len(OUTS[old_name].shape)==4 and self.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[old_name].shape[1], affine=False))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[old_name].shape[1], affine=False))
                        elif self.bn=='layerwise':
                                self.layers.add_module(ll['name'],nn.LayerNorm(OUTS[old_name].shape[2:4]))
                        elif self.bn=='instance':
                            self.layers.add_module(ll['name'], nn.InstanceNorm2d(OUTS[old_name].shape[1],affine=True))
                        elif self.bn=='simple':
                            self.layers.add_module(ll['name'],diag2d(OUTS[old_name].shape[1]))
                        else:
                            self.layers.add_module(ll['name'],Iden())
                        if self.back:
                            if self.bn=='full':
                                if len(OUTS[inp_ind].shape)==4 and self.bn:
                                    self.back_layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[inp_ind].shape[1]))
                                else:
                                    self.back_layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[inp_ind].shape[1]))
                            elif self.bn == 'half_full':
                                    if len(OUTS[old_name].shape) == 4 and self.bn:
                                        self.back_layers.add_module(ll['name'],
                                                           nn.BatchNorm2d(OUTS[inp_ind].shape[1], affine=False))
                                    else:
                                        self.back_layers.add_module(ll['name'],
                                                           nn.BatchNorm1d(OUTS[inp_ind].shape[1], affine=False))
                            elif self.bn == 'layerwise':
                                self.layers.add_module(ll['name'], nn.LayerNorm(OUTS[inp_ind].shape[2:4]))
                            elif self.bn == 'instance':
                                self.layers.add_module(ll['name'],
                                                       nn.InstanceNorm2d(OUTS[inp_ind].shape[1], affine=True))
                            elif self.bn == 'simple':
                                    self.back_layers.add_module(ll['name'], diag2d(OUTS[inp_ind].shape[1]))
                            else:
                                    self.back_layers.add_module(ll['name'], Iden())
                    if not self.first:
                        out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    else:
                        out = OUTS[inp_ind]
                    OUTS[ll['name']]=out

                if ('opr' in ll['name']):
                    if 'add' in ll['name']:
                        out = OUTS[inp_ind[0]]+OUTS[inp_ind[1]]
                        OUTS[ll['name']] = out
                        inp_feats=out.shape[1]
                if ('num_filters' in ll):
                    inp_feats = ll['num_filters']
                if ('shifts' in ll['name']):
                     inp_feats=OUTS[ll['name']].shape[1]
                if self.first:
                    self.fout.write(ll['name']+' '+str(np.array(OUTS[ll['name']].shape))+'\n')

                prev_shape=OUTS[ll['name']].shape
                in_dim=np.prod(OUTS[ll['name']].shape[1:])
                in_dims+=[in_dim]
                old_name=ll['name']


        if self.first==1:
            #print(self.layers, file=self.fout)
            tot_pars = 0
            KEYS=[]
            for keys, vals in self.named_parameters():
                if 'running' not in keys and 'tracked' not in keys:
                    KEYS+=[keys]
                #tot_pars += np.prod(np.array(vals.shape))

            # TEMPORARY
            pp=[]
            self.KEYS=KEYS
            for k,p in zip(KEYS,self.parameters()):
                if (self.update_layers is None):
                    if self.first==1:
                        self.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                    tot_pars += np.prod(np.array(p.shape))
                    pp+=[p]
                else:
                    found = False
                    for u in self.update_layers:
                        if u == k.split('.')[1]:
                            found=True
                            if self.first==1:
                                self.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                            tot_pars += np.prod(np.array(p.shape))
                            pp+=[p]
                    if not found:
                        p.requires_grad=False
            if self.first==1:
                self.fout.write('tot_pars,' + str(tot_pars)+'\n')
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
        if(everything or self.randomize is not None or self.penalize_activations is not None):
            out1=OUTS

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

    # GRADIENT STEP
    def loss_and_acc(self, input, target, dtype="train", lnum=0):


        # Embedding training with image and its deformed counterpart
        if type(input) is list:
            out0,ot0=self.forward(input[0])
            out1,ot1=self.forward(input[1])
            if self.embedd_type=='orig':
                loss, acc = get_embedd_loss(out0,out1,self.dv,self.no_standardize)
                #loss1, acc=simclr_loss(out0,out1,self.dv, self.no_standardize)
                #loss2=self.CLR(torch.cat((out0,out1),dim=0))
                #print(loss,loss2)
            elif self.embedd_type=='binary':
                loss, acc = get_embedd_loss_binary(out0,out1,self.dv,self.no_standardize)
            elif self.embedd_type=='L1dist_hinge':
                loss, acc = get_embedd_loss_new(out0,out1,self.dv,self.no_standardize, thr=self.thr, delta=self.delta)

        # Classification training
        else:
            if self.randomize is not None and dtype=="train":
                for i, k in enumerate(self.KEYS):
                    if self.randomize[lnum*2] not in k and self.randomize[lnum*2+1] not in k:
                        self.optimizer.param_groups[0]['params'][i].requires_grad=False
                    else:
                        self.optimizer.param_groups[0]['params'][i].requires_grad = True

            out,OUT=self.forward(input)
            if self.randomize is not None:
                out=OUT[self.randomize[lnum*2+1]]
            pen=0
            if self.penalize_activations is not None:
                for l in self.layer_text:
                    if 'penalty' in l:
                        pen+=self.penalize_activations*torch.sum(torch.mean((OUT[l['name']] * OUT[l['name']]).reshape(self.bsz,-1),dim=1))
            # Compute loss and accuracy
            loss, acc=self.get_acc_and_loss(out,target)
            loss+=pen
        return loss, acc



    # Epoch of network training
    def run_epoch(self, train, epoch, num_mu_iter=None, trainMU=None, trainLOGVAR=None, trPI=None, d_type='train', fout='OUT'):

        if (d_type=='train'):
            self.train()
        else:
            self.eval()
        num_tr=train[0].shape[0]
        ii = np.arange(0, num_tr, 1)
        if (d_type=='train'):
          np.random.shuffle(ii)
        jump = self.bsz
        trin = train[0][ii]
        targ = train[2][ii]
        self.n_class = np.max(targ) + 1

        ll=1
        if self.randomize is not None:
            ll=len(self.randomize)//2
            nrep=(num_tr//jump)//ll
            lnums=np.repeat(np.array(range(ll)),nrep)
            np.random.shuffle(lnums)

        full_loss=np.zeros(ll); full_acc=np.zeros(ll); count=np.zeros(ll)

        # Loop over batches.

        targ_in=targ
        TIME=0
        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            lnum=0
            if self.randomize:
                lnum=lnums[j//jump]
            if (d_type == 'train'):
                self.optimizer.zero_grad()
            if self.embedd:
                with torch.no_grad():
                    data_in=torch.from_numpy(trin[j:j + jump]).float()
                    data_out1=deform_data(data_in,self.perturb,self.trans,self.s_factor,self.h_factor,self.embedd)
                    data=[data_in.to(self.dv),data_out1.to(self.dv)]
            else:
                dd=torch.from_numpy(trin[j:j + jump])

                if self.perturb>0.and d_type=='train':
                   with torch.no_grad():
                     dd = deform_data(dd, self.perturb, self.trans, self.s_factor, self.h_factor,self.embedd)
                data = dd.to(self.dv,dtype=torch.float32)


            target = torch.from_numpy(targ_in[j:j + jump]).to(self.dv, dtype=torch.long)

            with torch.no_grad() if (d_type!='train') else dummy_context_mgr():
                loss, acc = self.loss_and_acc(data, target,dtype=d_type, lnum=lnum)
            if (d_type == 'train'):
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip>0.:
                    nn.utils.clip_grad_value_(self.parameters(),self.grad_clip)


                self.optimizer.step()
                if 'xla' in self.dv.type:
                    xm.mark_step()
                #if self.scheduler is not None:
                #  self.scheduler.step()


            full_loss[lnum] += loss.item()
            full_acc[lnum] += acc.item()
            count[lnum]+=1


        for l in range(ll):
            fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.4F} \n'.format(d_type,epoch,
                    full_loss[l] /count[l], full_acc[l]/(count[l]*jump)))

        return trainMU, trainLOGVAR, trPI, [full_acc/(count*jump), full_loss/(count)]

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
        if args.sched[0] > 0:
            lambda1 = lambda epoch: args.sched[1]**(epoch // np.int32(args.sched[0]))
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda1)
            #self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[50,100,150,200,250,300,350],args.sched)
            #l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), args.sched)
            #scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)





