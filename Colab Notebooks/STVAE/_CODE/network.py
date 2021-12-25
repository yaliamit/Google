
from torch import nn, optim
import contextlib
from images import deform_data
from losses import *
import sys
from layers import *
import time
from data import DL

try:
    import torch_xla.core.xla_model as xm
except:
    pass
@contextlib.contextmanager
def dummy_context_mgr():
    yield None

pre=get_pre()

osu=os.uname()

def get_criterion(args):
    if args.hinge:
        args.temp.criterion = hinge_loss(num_class=args.num_class)
    else:
        args.temp.criterion = nn.CrossEntropyLoss()
    args.CLR = SimCLRLoss(args.mb_size, args.temp.dv)

class temp_args(nn.Module):
    def  __init__(self):
        super(temp_args,self).__init__()
        self.back=None
        self.first=0
        self.everything=False
        self.lnti = None
        self.layer_text = None
        self.dv = None
        self.optimizer=None
        self.embedd_layer=None
        KEYS=None
    

def initialize_model(model,args, sh,lnti,layers_dict,device):

    args.temp=temp_args()
    args.temp.lnti=lnti
    args.temp.layer_text=layers_dict
    args.temp.dv=device
    args.temp.back = ('ae' in args.type)
    args.temp.everything=False
    get_criterion(args)
    if args.crop and len(sh) == 3:
        sh = (sh[0], args.crop, args.crop)
        print(sh)

    if args.clapp_dim is not None:
        model.add_module('clapp', nn.Conv2d(args.clapp_dim[1], args.clapp_dim[1], 1))

    if sh is not None:
        temp = torch.zeros([1] + list(sh))  # .to(device)
        # Run the network once on dummy data to get the correct dimensions.
        args.temp.first=1
        bb = model.forward(temp,args)
        args.temp.output_shape = bb[0].shape


        if args.temp.first==1:
            #print(self.layers, file=self.fout)
            tot_pars = 0
            KEYS=[]
            for keys, vals in model.named_parameters():
                if 'running' not in keys and 'tracked' not in keys:
                    KEYS+=[keys]
                #tot_pars += np.prod(np.array(vals.shape))

            # TEMPORARY
            if True:
                pp=[]
                args.temp.KEYS=KEYS
                for k,p in zip(KEYS,model.parameters()):
                    if (args.update_layers is None):
                        if args.temp.first==1:
                            args.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                        tot_pars += np.prod(np.array(p.shape))
                        pp+=[p]
                    else:
                        found = False
                        for u in args.update_layers:
                            if u == k.split('.')[1] or u==k.split('.')[0]:
                                found=True
                                if args.temp.first==1:
                                    args.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                                tot_pars += np.prod(np.array(p.shape))
                                pp+=[p]
                        if not found:
                            p.requires_grad=False
                if args.temp.first==1:
                    args.fout.write('tot_pars,' + str(tot_pars)+'\n')
                if 'ae' not in args.type:
                    if (args.optimizer_type == 'Adam'):
                            if args.temp.first==1:
                                args.fout.write('Optimizer Adam '+str(args.lr)+'\n')
                            args.temp.optimizer = optim.Adam(pp, lr=args.lr,weight_decay=args.wd)
                    else:
                            if args.first==1:
                                args.fout.write('Optimizer SGD '+str(args.lr))
                            args.temp.optimizer = optim.SGD(pp, lr=args.lr,weight_decay=args.wd)

            args.temp.first=0
        #model.add_module('temp',args.temp)
        bsz=args.mb_size
        if args.use_multiple_gpus is not None:
            bsz=bsz//args.use_multiple_gpus
        args.temp.loss=L1_loss(args.temp.dv, bsz)
        #model.to(args.temp.dv)


def get_acc_and_loss(args, out, targ):
            v, mx = torch.max(out, 1)
            # Non-space characters
            # Total loss
            loss = args.temp.criterion(out, targ)
            # total accuracy
            acc = torch.sum(mx.eq(targ))
            return loss, acc



# Network module
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()



    def forward(self,input,args,clapp=False, lay=None):

        #print('IN',input.shape, input.get_device())
        #if args.temp.first==0:
         #   args.temp=self.temp
            #print('INP_dim',input.shape[0])
        out = input
        in_dims=[]
        if (args.temp.first):
            self.layers = nn.ModuleList()
            if args.temp.back:
                self.back_layers=nn.ModuleList()
        OUTS={}
        old_name=''

        DONE=False
        for i,ll in enumerate(args.temp.layer_text):
            if not DONE:
                inp_ind = old_name

                if ('parent' in ll):
                    pp=ll['parent']
                    # over ride default inp_feats
                    if len(pp)==1:
                        inp_ind=pp[0]
                        if args.temp.first:
                            inp_feats=OUTS[pp[0]].shape[1]
                            in_dim=in_dims[args.temp.lnti[pp[0]]]
                    else:
                        inp_feats=[]
                        loc_in_dims=[]
                        inp_ind=[]
                        for p in pp:
                            inp_ind += [p]
                            if args.first:
                                inp_feats+=[OUTS[p].shape[1]]
                                loc_in_dims+=[in_dims[args.lnti[p]]]
                if ('input' in ll['name']):
                    OUTS[ll['name']]=input
                if ('shift' in ll['name']):
                     if args.temp.first:
                         self.layers.add_module(ll['name'],shifts(ll['shifts']))
                     out=getattr(self.layers,ll['name'])(OUTS[inp_ind])
                     OUTS[ll['name']]=out
                if ('conv' in ll['name']):
                    if args.temp.first:
                        bis = True
                        if ('nb' in ll):
                            bis = False
                        stride=1;
                        if 'stride' in ll:
                            stride=ll['stride']
                        pd=(ll['filter_size']//stride) // 2
                        if not args.temp.back:
                            if 'fa' in ll['name'] and not 'ga' in pre and 'Darwin' not in os.uname():
                                self.layers.add_module(ll['name'],FAConv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,fa=args.fa,padding=pd, bias=bis))
                            else:
                                self.layers.add_module(ll['name'],nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,padding=pd, bias=bis))
                            if 'zero' in ll:
                                temp=getattr(self.layers, ll['name'])
                                temp.weight.data=ll['zero']*torch.ones_like(temp.weight.data)
                        else:
                            self.layers.add_module(ll['name'],nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=1,padding=pd))
                            self.back_layers.add_module(ll['name']+'_bk',nn.Conv2d(ll['num_filters'],inp_feats,ll['filter_size'],stride=1,padding=pd))
                            if 'zero' in ll:
                                temp=getattr(self.back_layers, ll['name']+'_bk')
                                temp.weight.data[0,0]=ll['zero']*torch.ones_like(temp.weight.data[0,0])
                                
                    out=getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out
                if 'non_linearity' in ll['name']:
                    if args.temp.first:
                        low=-1.; high=1.
                        if 'lims' in ll:
                            low=ll['lims'][0]; high=ll['lims'][1]
                        self.layers.add_module(ll['name'],NONLIN(ll['type'],low=low,high=high))
                        if args.temp.back:
                            self.back_layers.add_module(ll['name']+'_bk',NONLIN(ll['type'],low=low,high=high))

                    OUTS[ll['name']] = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                if ('Avg' in ll['name']):
                    if args.first:
                        HW=(np.int32(OUTS[inp_ind].shape[2]/2),np.int32(OUTS[inp_ind].shape[3]/2))
                        self.layers.add_module(ll['name'],nn.AvgPool2d(HW,HW))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']] = out
                if ('pool' in ll['name']):
                    if args.temp.first:
                        stride = ll['pool_size']
                        if ('stride' in ll):
                            stride = ll['stride']
                        pp=[np.int32(np.mod(ll['pool_size'],2))]
                        pp=(ll['pool_size']-1)//2
                        self.layers.add_module(ll['name'],nn.MaxPool2d(ll['pool_size'], stride=stride, padding=pp))
                        if args.temp.back:
                            #self.back_layers.add_module(ll['name']+'_bk',nn.UpsamplingNearest2d(scale_factor=stride))
                            self.back_layers.add_module(ll['name']+'_bk',nn.UpsamplingNearest2d(size=out.shape[2:4]))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out


                if ('drop' in ll['name']):
                    if args.temp.first:
                        self.layers.add_module(ll['name'],torch.nn.Dropout(p=ll['drop'], inplace=False))
                        if args.temp.back:
                            self.back_layers.add_module(ll['name']+'_bk',torch.nn.Dropout(p=ll['drop'],inplace=False))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']]=out

                if ('dense' in ll['name']):
                    if args.temp.first:
                        out_dim=ll['num_units']
                        bis=True
                        if ('nb' in ll):
                            bis=False
                        if not args.temp.back:
                            if 'fa' in ll['name']:
                                self.layers.add_module(ll['name'],FALinear(in_dim,out_dim,bias=bis, fa=args.fa))
                            else:
                                self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis))
                        else:
                            self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis))

                        if args.temp.back:
                            self.back_layers.add_module(ll['name']+'_bk_'+'reshape',Reshape(list(OUTS[inp_ind].shape[1:])))
                            self.back_layers.add_module(ll['name']+'_bk',nn.Linear(out_dim,in_dim,bias=bis))
                    out=OUTS[inp_ind]
                    out = out.reshape(out.shape[0], -1)
                    out=getattr(self.layers, ll['name'])(out)
                    OUTS[ll['name']]=out
                if 'inject' in ll['name']:
                    if args.temp.first:
                        stride=ll['stride']
                        self.layers.add_module(ll['name'],Inject(stride))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']] = out
                if 'subsample' in ll['name']:
                    if args.temp.first:
                        stride = None
                        if 'stride' in ll:
                            stride = ll['stride']
                        self.layers.add_module(ll['name'], Subsample(stride=stride))
                        if args.temp.back:
                            #self.back_layers.add_module(ll['name']+'_bk',nn.UpsamplingNearest2d(scale_factor=stride))
                            self.back_layers.add_module(ll['name']+'_bk',Inject(stride))
                    out = getattr(self.layers, ll['name'])(OUTS[inp_ind])
                    OUTS[ll['name']] = out
                if ('norm') in ll['name']:
                    if args.temp.first:
                        if args.bn=='full':
                            if len(OUTS[old_name].shape)==4 and args.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[old_name].shape[1]))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[old_name].shape[1]))
                        elif args.bn=='half_full':
                            if len(OUTS[old_name].shape)==4 and args.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[old_name].shape[1], affine=False))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[old_name].shape[1], affine=False))
                        elif args.bn=='layerwise':
                                self.layers.add_module(ll['name'],nn.LayerNorm(OUTS[old_name].shape[2:4]))
                        elif args.bn=='instance':
                            self.layers.add_module(ll['name'], nn.InstanceNorm2d(OUTS[old_name].shape[1],affine=True))
                        elif args.bn=='simple':
                            self.layers.add_module(ll['name'],diag2d(OUTS[old_name].shape[1]))
                        else:
                            self.layers.add_module(ll['name'],Iden())
                        if args.temp.back:
                            if args.bn=='full':
                                if len(OUTS[inp_ind].shape)==4 and args.bn:
                                    self.back_layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[inp_ind].shape[1]))
                                else:
                                    self.back_layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[inp_ind].shape[1]))
                            elif args.bn == 'half_full':
                                    if len(OUTS[old_name].shape) == 4 and args.bn:
                                        self.back_layers.add_module(ll['name'],
                                                           nn.BatchNorm2d(OUTS[inp_ind].shape[1], affine=False))
                                    else:
                                        self.back_layers.add_module(ll['name'],
                                                           nn.BatchNorm1d(OUTS[inp_ind].shape[1], affine=False))
                            elif args.bn == 'layerwise':
                                self.layers.add_module(ll['name'], nn.LayerNorm(OUTS[inp_ind].shape[2:4]))
                            elif args.bn == 'instance':
                                self.layers.add_module(ll['name'],
                                                       nn.InstanceNorm2d(OUTS[inp_ind].shape[1], affine=True))
                            elif args.bn == 'simple':
                                    self.back_layers.add_module(ll['name'], diag2d(OUTS[inp_ind].shape[1]))
                            else:
                                    self.back_layers.add_module(ll['name'], Iden())
                    if not args.temp.first:
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
                if args.temp.first:
                    args.fout.write(ll['name']+' '+str(np.array(OUTS[ll['name']].shape))+'\n')

                prev_shape=OUTS[ll['name']].shape
                in_dim=np.prod(OUTS[ll['name']].shape[1:])
                in_dims+=[in_dim]
                old_name=ll['name']
                if lay is not None and lay in ll['name']:
                    DONE=True

        if args.embedd_type == 'clapp':

            if args.temp.first:
                args.clapp_dim = prev_shape
                self.add_module('clapp', nn.Conv2d(args.clapp_dim[1], args.clapp_dim[1], 1))
                if args.update_layers is not None:
                    args.update_layers.append('clapp')
            if clapp:
                out=self.clapp(OUTS[old_name])

        out1=[]

        if(args.temp.everything or args.randomize is not None or args.penalize_activations is not None):
            out1=OUTS

        return(out,out1)

    def backwards(self,x):
        xx=x
        for l in reversed(list(self.back_layers)):
            xx=l(xx)

        return xx



def run_epoch(model, args, train, epoch, d_type='train', fout='OUT',freq=1):

        if (d_type=='train'):
            model.train()
        else:
            model.eval()

        #if type(train) is DL:
        jump=train.batch_size
        args.n_class=train.num_class
        num_tr=train.num

        ll=1
        full_loss=np.zeros(ll); full_acc=np.zeros(ll); count=np.zeros(ll)

        # Loop over batches.

        #if isinstance(model, torch.nn.DataParallel):
        #    dvv=model.module.temp.dv
        #    optimizer=model.module.temp.optimizer
        #else:
        #    dvv=model.temp.dv
        #    optimizer=model.temp.optimizer
        optimizer=args.temp.optimizer
        dvv=args.temp.dv
        TIME=0
        tra=iter(train)
        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            lnum=0
            if d_type=='train':
                optimizer.zero_grad()
            #if type(train) is DL:
            BB, indlist=next(tra)
            data_in=BB[0].to(dvv,non_blocking=True)
            target=BB[1].to(dvv, dtype=torch.long)


            if args.embedd:

                with torch.no_grad():
                    if args.crop==0:
                        data_out=deform_data(data_in,args.perturb,args.transformation,args.s_factor,args.h_factor, args.embedd,dvv)
                        data_in=deform_data(data_in,args.perturb,args.transformation,args.s_factor,args.h_factor,args.embedd,dvv)
                        data=[data_in,data_out]
                    else:
                        data_p=data_in
                        data=[data_p[0],data_p[1]]
            else:
                if args.perturb>0.and d_type=='train':
                   with torch.no_grad():
                     data_in = deform_data(data_in, args.perturb, args.transformation, args.s_factor, args.h_factor,args.embedd, dvv)
                data = data_in#.to(model.temp.dv,dtype=torch.float32)



            with torch.no_grad() if (d_type!='train') else dummy_context_mgr():
                loss, acc = loss_and_acc(model, args, data, target,dtype=d_type, lnum=lnum)
            if (d_type == 'train'):
                loss.backward()
                if args.grad_clip>0.:
                    nn.utils.clip_grad_value_(model.parameters(),args.grad_clip)
                optimizer.step()

            full_loss[lnum] += loss.item()
            full_acc[lnum] += acc.item()
            count[lnum]+=1


        if freq-np.mod(epoch,freq)==1:
           for l in range(ll):
                fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.6F} \n'.format(d_type,epoch,
                    full_loss[l] /count[l], full_acc[l]/(count[l]*jump)))

        return [full_acc/(count*jump), full_loss/(count)]


def loss_and_acc(model, args, input, target, dtype="train", lnum=0):

        # if isinstance(model, torch.nn.DataParallel):
        #     dvv = model.module.temp.dv
        #     optimizer = model.module.temp.optimizer
        # else:
        #     dvv = model.temp.dv
        #     optimizer = model.temp.optimizer
        dvv=args.temp.dv
        optimizer=args.temp.optimizer
        # Embedding training with image and its deformed counterpart
        if type(input) is list:

            out1, ot1 = model.forward(input[1], args)
            # print('out1',out1.device.index)
            with torch.no_grad():
                cl = False
                if args.embedd_type == 'clapp':
                    cl = True
                out0, ot0 = model.forward(input[0], args, clapp=cl)
                # print('out0', out0.device.index)
            if args.embedd_type == 'orig':
                pass
                # loss, acc = get_embedd_loss(out0,out1,dvv,args.thr)
            elif args.embedd_type == 'binary':
                pass
                # loss, acc = get_embedd_loss_binary(out0,out1,dvv,args.no_standardize)
            elif args.embedd_type == 'L1dist_hinge':
                loss, acc = args.temp.loss(out0, out1, dvv, args.no_standardize, future=args.future, thr=args.thr,
                                           delta=args.delta)
            elif args.embedd_type == 'clapp':
                pass
                # out0 = out0.reshape(out0.shape[0], -1)
                # out1 = out1.reshape(out1.shape[0], -1)
                # loss, acc = get_embedd_loss_clapp(out0,out1,dvv,args.thr)
        # Classification training

        else:

            if args.randomize_layers is not None and dtype == "train":
                for i, k in enumerate(args.KEYS):
                    if args.randomize_layers[lnum * 2] not in k and args.randomize_layers[lnum * 2 + 1] not in k:
                        optimizer.param_groups[0]['params'][i].requires_grad = False
                    else:
                        optimizer.param_groups[0]['params'][i].requires_grad = True

            out, OUT = model.forward(input, args)
            if args.randomize_layers is not None:
                out = OUT[args.randomize_layers[lnum * 2 + 1]]
            pen = 0
            if args.penalize_activations is not None:
                for l in args.layer_text:
                    if 'penalty' in l:
                        pen += args.penalize_activations * torch.sum(
                            torch.mean((OUT[l['name']] * OUT[l['name']]).reshape(args.mb_size, -1), dim=1))
            # Compute loss and accuracy
            loss, acc = get_acc_and_loss(args, out, target)
            loss += pen
            #if dtype=='train':
             #   loss.backward()
        return loss, acc

def get_embedding(model, args, train):


        jump = train.batch_size
        num_tr = train.num
        args.everything=True
        model.eval()
        OUT=[]
        labels=[]
        tra=iter(train)

        for j in np.arange(0, num_tr, jump, dtype=np.int32):
            BB = next(tra)
            data = BB[0]
            labels+=[BB[1].numpy()]
            data=data.to(model.temp.dv)
            with torch.no_grad():
                out=model.forward(data, args, lay=args.embedd_layer)[1][args.embedd_layer].detach().cpu().numpy()
                OUT+=[out]

        OUTA=np.concatenate(OUT,axis=0)
        labels=np.concatenate(labels)

        return [OUTA,labels]

def get_scheduler(args,optimizer):
        scheduler = None
        if args.sched[0] > 0:
            lambda1 = lambda epoch: args.sched[1]**(epoch // np.int32(args.sched[0]))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1)
            #self.scheduler=torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[50,100,150,200,250,300,350],args.sched)
            #l2 = lambda epoch: pow((1. - 1. * epoch / args.nepoch), args.sched)
            #scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=l2)

        return scheduler


