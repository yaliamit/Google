
from torch import nn, optim
import contextlib
from images import deform_data
from losses import *
import sys
from layers import *
from get_net_text import get_network
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


class temp_args(nn.Module):
    def  __init__(self):
        super(temp_args,self).__init__()
        self.back=None
        self.first=0
        self.everything=False
        self.layer_text = None
        self.dv = None
        self.optimizer=None
        self.embedd_layer=None
        KEYS=None
    

def initialize_model(args, sh, layers,device, layers_dict=None):

    model=network()
    #if not read_model:
    if layers_dict==None:
            layers_dict=get_network(layers)
    # else:
    #     print('LOADING OLD MODEL')
    #     sm = torch.load(datadirs + '_output/' + args.model + '.pt', map_location='cpu')
    #     args=sm['args']
    #     # model_old = network.network()
    for l in layers_dict:
        if 'dense_gaus' in l['name']:
            l['num_units']=sh[0]
    atemp = temp_args()
    atemp.layer_text = layers_dict
    atemp.dv = device
    atemp.everything = False
    atemp.bn=args.bn
    atemp.fout=args.fout
    atemp.fa=args.fa
    atemp.embedd_type=args.embedd_type
    atemp.randomize_layers=args.randomize_layers
    atemp.penalize_activations=args.penalize_activations
    if args.hinge:
        atemp.loss = hinge_loss(num_class=args.num_class)
    else:
        atemp.loss = nn.CrossEntropyLoss()

    if args.crop and len(sh) == 3:
        sh = (sh[0], args.crop, args.crop)
        print(sh)
    if args.embedd_type=='clapp' and args.embedd:
        if args.clapp_dim is not None:
            model.add_module('clapp', nn.Conv2d(args.clapp_dim[1], args.clapp_dim[1], 1))
        if args.update_layers is not None:
            args.update_layers.append('clapp')

    if sh is not None:
        temp = torch.zeros([1] + list(sh))  # .to(device)
        # Run the network once on dummy data to get the correct dimensions.
        atemp.first=1
        atemp.input_shape=None
        bb = model.forward(temp,atemp)
        if args.embedd_type=='clapp' and args.embedd:
            args.clapp_dim=atemp.clapp_dim

        atemp.output_shape = bb[0].shape
        if atemp.input_shape is None:
            atemp.input_shape = sh

        if 'ae' not in args.type:
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
                atemp.KEYS=KEYS
                for k,p in zip(KEYS,model.parameters()):
                    if (args.update_layers is None):
                        if atemp.first==1:
                            atemp.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                        tot_pars += np.prod(np.array(p.shape))
                        pp+=[p]
                    else:
                        found = False
                        for u in args.update_layers:
                            if u == k.split('.')[1] or u==k.split('.')[0]:
                                found=True
                                if atemp.first==1:
                                    atemp.fout.write('TO optimizer '+k+ str(np.array(p.shape))+'\n')
                                tot_pars += np.prod(np.array(p.shape))
                                pp+=[p]
                        if not found:
                            p.requires_grad=False
                if atemp.first==1:
                    atemp.fout.write('tot_pars,' + str(tot_pars)+'\n')
                if 'ae' not in args.type:
                    if (args.optimizer_type == 'Adam'):
                            if atemp.first==1:
                                atemp.fout.write('Optimizer Adam '+str(args.lr)+'\n')
                            atemp.optimizer = optim.Adam(pp, lr=args.lr,weight_decay=args.wd)
                    else:
                            if atemp.first==1:
                                atemp.fout.write('Optimizer SGD '+str(args.lr))
                            atemp.optimizer = optim.SGD(pp, lr=args.lr,weight_decay=args.wd)

        atemp.first=0
        bsz=args.mb_size
        if args.embedd:
            if args.embedd_type=='L1dist_hinge':
                atemp.loss=L1_loss(atemp.dv, bsz, args.future, args.thr, args.delta, WW=1., nostd=True)
            elif args.embedd_type=='clapp':
                atemp.loss=clapp_loss(atemp.dv)
            elif args.embedd_type=='binary':
                atemp.loss=binary_loss(atemp.dv)
            elif args.embedd_type=='direct':
                atemp.loss=direct_loss(bsz,atemp.output_shape[1],device=atemp.dv)
            elif args.embedd_type=='orig':
                atemp.loss=SIMCLR_loss(atemp.dv)

        model.add_module('temp',atemp)
        if args.use_multiple_gpus is not None:
            bsz=bsz//args.use_multiple_gpus
        model=model.to(atemp.dv)
        return model


def get_acc_and_loss(aloss, out, targ):
            v, mx = torch.max(out, 1)
            # Non-space characters
            # Total loss
            loss = aloss(out, targ)
            # total accuracy        out = input
            acc = torch.sum(mx.eq(targ))
            return loss, acc



# Network module
class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()



    def forward(self,input,atemp=None,clapp=False, lay=None):

        #print('IN',input.shape, input.get_device())
        if atemp is None:
            atemp=self.temp

        everything = atemp.first or atemp.everything or atemp.randomize_layers is not None or atemp.penalize_activations is not None or lay is not None

            #print('INP_dim',input.shape[0])
        in_dims={}
        if (atemp.first):
            self.layers = nn.ModuleList()

        OUTS={}
        old_name=''

        DONE=False
        for i,ll in enumerate(atemp.layer_text):
            if not DONE:
                inp_ind = old_name

                if ('parent' in ll):
                    pp=ll['parent']
                    # over ride default inp_feats
                    if len(pp)==1:
                        inp_ind=pp[0]
                        if atemp.first:
                            inp_feats=OUTS[pp[0]].shape[1]
                            in_dim=in_dims[pp[0]]
                    else:
                        inp_feats=[]
                        loc_in_dims=[]
                        inp_ind=[]
                        for p in pp:
                            inp_ind += [p]
                            if atemp.first:
                                inp_feats+=[OUTS[p].shape[1]]
                                loc_in_dims+=[in_dims[p]]
                if ('input' in ll['name']):
                    out=input
                    if atemp.first:
                        if 'shape' in ll and 'num_filters' in ll:
                            atemp.input_shape=[ll['num_filters']]+list(ll['shape'])

                     #   out=out.reshape(out.shape[0],ll['num_filters'],)
                    if everything:
                        OUTS[ll['name']]=out

                if ('shift' in ll['name']):
                     if atemp.first:
                         self.layers.add_module(ll['name'],shifts(ll['shifts']))
                     out=getattr(self.layers,ll['name'])(out)
                     if everything:
                         OUTS[ll['name']] = out
                if ('conv' in ll['name']):
                    if everything:
                        out = OUTS[inp_ind]
                    if len(out.shape)==2:
                        wdim=np.int(np.sqrt(out.shape[1]/inp_feats))
                        out=out.reshape(out.shape[0],inp_feats,wdim,wdim)
                    if atemp.first:
                        bis = True
                        if ('nb' in ll):
                            bis = False
                        stride=1;
                        if 'stride' in ll:
                            stride=ll['stride']
                        pd=(ll['filter_size']//stride) // 2
                        if 'fa' in ll['name'] and not 'ga' in pre and 'Darwin' not in os.uname():
                                self.layers.add_module(ll['name'],FAConv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,fa=atemp.fa,padding=pd, bias=bis))
                        else:
                                self.layers.add_module(ll['name'],nn.Conv2d(inp_feats,ll['num_filters'],ll['filter_size'],stride=stride,padding=pd, bias=bis))
                        if 'zero' in ll:
                                temp=getattr(self.layers, ll['name'])
                                temp.weight.data=ll['zero']*torch.ones_like(temp.weight.data)

                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if 'non_linearity' in ll['name']:
                    if atemp.first:
                        low=-1.; high=1.
                        if 'lims' in ll:
                            low=ll['lims'][0]; high=ll['lims'][1]
                        self.layers.add_module(ll['name'],NONLIN(ll['type'],low=low,high=high))

                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out
                if ('Avg' in ll['name']):
                    if atemp.first:
                        HW=(np.int32(OUTS[inp_ind].shape[2]/2),np.int32(OUTS[inp_ind].shape[3]/2))
                        self.layers.add_module(ll['name'],nn.AvgPool2d(HW,HW))
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('pool' in ll['name']):
                    if atemp.first:
                        stride = ll['pool_size']
                        if ('stride' in ll):
                            stride = ll['stride']
                        pp=[np.int32(np.mod(ll['pool_size'],2))]
                        pp=(ll['pool_size']-1)//2
                        self.layers.add_module(ll['name'],nn.MaxPool2d(ll['pool_size'], stride=stride, padding=pp))


                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('drop' in ll['name']):
                    if atemp.first:
                        self.layers.add_module(ll['name'],torch.nn.Dropout(p=ll['drop'], inplace=False))


                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('dense' in ll['name']):
                    if atemp.first:
                        out_dim=ll['num_units']
                        bis=True
                        if ('nb' in ll):
                            bis=False
                        if 'fa' in ll['name']:
                                self.layers.add_module(ll['name'],FALinear(in_dim,out_dim,bias=bis, fa=atemp.fa))
                        else:
                            if 'Lin' in ll:
                                self.layers.add_module(ll['name'],Linear(in_dim,out_dim, scale=0, iden=True))
                            else:
                                self.layers.add_module(ll['name'],nn.Linear(in_dim,out_dim,bias=bis))
                    if everything:
                        out=OUTS[inp_ind]
                    out = out.reshape(out.shape[0], -1)
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if 'inject' in ll['name']:
                    if atemp.first:
                        self.layers.add_module(ll['name'],Inject(ll))
                    if everything:
                        out = OUTS[inp_ind]
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if 'subsample' in ll['name']:
                    if atemp.first:
                        stride = None
                        if 'stride' in ll:
                            stride = ll['stride']
                        self.layers.add_module(ll['name'], Subsample(stride=stride))

                    if everything:
                        out = OUTS[inp_ind]
                    out = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('norm') in ll['name']:
                    if atemp.first:
                        if atemp.bn=='full':
                            if len(OUTS[old_name].shape)==4 and atemp.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[old_name].shape[1]))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[old_name].shape[1]))
                        elif atemp.bn=='half_full':
                            if len(OUTS[old_name].shape)==4 and atemp.bn:
                                self.layers.add_module(ll['name'],nn.BatchNorm2d(OUTS[old_name].shape[1], affine=False))
                            else:
                                self.layers.add_module(ll['name'],nn.BatchNorm1d(OUTS[old_name].shape[1], affine=False))
                        elif atemp.bn=='layerwise':
                                self.layers.add_module(ll['name'],nn.LayerNorm(OUTS[old_name].shape[2:4]))
                        elif atemp.bn=='instance':
                            self.layers.add_module(ll['name'], nn.InstanceNorm2d(OUTS[old_name].shape[1],affine=True))
                        elif atemp.bn=='simple':
                            self.layers.add_module(ll['name'],diag2d(OUTS[old_name].shape[1]))
                        else:
                            self.layers.add_module(ll['name'],Iden())

                    if not atemp.first:
                        out = getattr(self.layers, ll['name'])(out)
                        if everything:
                            OUTS[ll['name']] = out
                    else:
                        pass


                if ('opr' in ll['name']):
                    if 'add' in ll['name']:
                        out = OUTS[inp_ind[0]]+OUTS[inp_ind[1]]
                        OUTS[ll['name']] = out
                        inp_feats=out.shape[1]
                if ('num_filters' in ll):
                    inp_feats = ll['num_filters']
                if ('shifts' in ll['name']):
                     inp_feats=out.shape[1]

                prev_shape=out.shape

                if atemp.first:
                    atemp.fout.write(ll['name']+' '+str(np.array(prev_shape))+'\n')

                in_dim=np.prod(prev_shape[1:])
                in_dims[ll['name']]=in_dim
                old_name=ll['name']
                if lay is not None and lay in ll['name']:
                    DONE=True

        if atemp.embedd_type == 'clapp':

            if atemp.first:
                atemp.clapp_dim = prev_shape
                self.add_module('clapp', nn.Conv2d(atemp.clapp_dim[1], atemp.clapp_dim[1], 1))

            if clapp:
                out = self.clapp(out)

        out1=[]

        if(everything):
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
        if 'ga' in get_pre() and args.use_multiple_gpus is not None:
            optimizer = model.module.temp.optimizer
            dvv=model.module.temp.dv
            lossf=model.module.temp.loss
        else:
            optimizer = model.temp.optimizer
            dvv=model.temp.dv
            lossf=model.temp.loss
        TIME=0
        tra=iter(train)
        out_norm=0
        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            lnum=0
            if d_type=='train':
                optimizer.zero_grad()

            BB, indlist=next(tra)
            data_in=BB[0].to(dvv,non_blocking=True)
            target=BB[1].to(dvv, dtype=torch.long)

            data=get_data(data_in,args,dvv, d_type)

            with torch.no_grad() if (d_type!='train') else dummy_context_mgr():
                out, OUT=forw(model,args,data)
                if model.temp.embedd_type=='direct':
                    out_norm+=torch.mean(torch.norm(out[0],dim=1))
                loss, acc = get_loss(lossf,args, out, OUT, target)
            if args.randomize_layers is not None and d_type == "train":
                    for i, k in enumerate(args.KEYS):
                        if args.randomize_layers[lnum * 2] not in k and args.randomize_layers[lnum * 2 + 1] not in k:
                            optimizer.param_groups[0]['params'][i].requires_grad = False
                        else:
                            optimizer.param_groups[0]['params'][i].requires_grad = True
            if (d_type == 'train'):
                loss.backward()
                if args.grad_clip>0.:
                    nn.utils.clip_grad_value_(model.parameters(),args.grad_clip)
                optimizer.step()

            full_loss[lnum] += loss.item()
            if acc is not None:
                full_acc[lnum] += acc.item()
            count[lnum]+=1


        if freq-np.mod(epoch,freq)==1:
           print('OUT NORM',out_norm/count[0])
           for l in range(ll):
                fout.write('\n ====> Ep {}: {} Full loss: {:.4F}, Full acc: {:.6F} \n'.format(d_type,epoch,
                    full_loss[l] /count[l], full_acc[l]/(count[l]*jump)))

        return [full_acc/(count*jump), full_loss/(count)]


def get_data(data_in, args, dvv, d_type):
    if args.embedd:
        with torch.no_grad():
            if args.crop == 0:
                data_out = deform_data(data_in, args.perturb, args.transformation, args.s_factor, args.h_factor,
                                       args.embedd, dvv)
                data_in = deform_data(data_in, args.perturb, args.transformation, args.s_factor, args.h_factor,
                                      args.embedd, dvv)
                data = [data_in, data_out]
            else:
                data_p = data_in
                data = [data_p[0], data_p[1]]
    else:
        if args.perturb > 0. and d_type == 'train':
            with torch.no_grad():
                data_in = deform_data(data_in, args.perturb, args.transformation, args.s_factor, args.h_factor,
                                      args.embedd, dvv)
        data = data_in
    return data

def forw(model, args, input, lnum=0):

    OUT=None
    if type(input) is list:

        out1, _ = model.forward(input[1])
        # print('out1',out1.device.index)
        with torch.no_grad():
            cl = False
            if args.embedd_type == 'clapp':
                cl = True
            out0, _ = model.forward(input[0], clapp=cl)
        out=[out0,out1]
    else:
        out, OUT = model.forward(input)
        if args.randomize_layers is not None:
            out = OUT[args.randomize_layers[lnum * 2 + 1]]

    return out, OUT

def get_loss(aloss, args, out, OUT, target):

        # Embedding training with image and its deformed counterpart
        if type(out) is list:
            loss,acc = aloss(out[0],out[1])
        else:

            pen = 0
            if args.penalize_activations is not None:
                for l in args.layer_text:
                    if 'penalty' in l:
                        pen += args.penalize_activations * torch.sum(
                            torch.mean((OUT[l['name']] * OUT[l['name']]).reshape(args.mb_size, -1), dim=1))
            # Compute loss and accuracy
            loss, acc = get_acc_and_loss(aloss, out, target)
            loss += pen

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
            BB, _ = next(tra)
            data = BB[0]
            labels+=[BB[1].numpy()]
            data=data.to(model.temp.dv)
            with torch.no_grad():
                out=model.forward(data, lay=args.embedd_layer)[1][args.embedd_layer].detach().cpu().numpy()
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


