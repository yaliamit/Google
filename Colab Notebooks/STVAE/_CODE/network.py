
from torch import nn, optim
import contextlib
from images import deform_data
from losses import *
import sys
from layers import *
import normflows as nf

import platform
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

osu=platform.system()

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

    def process_nfl(self, ll):
        nu = ll['num_units']
        ni = ll['num_inputs']
        param_map = nf.nets.MLP([ni, nu, nu, 2 * ni], init_zeros=True)
        # Add flow layer
        self.layers.add_module(ll['name'], nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        self.layers.add_module(ll['name'] + 'perm', nf.flows.Permute(2 * ni, mode='shuffle'))

    def process_dense(self, in_dim, ll):
        out_dim = ll['num_units']
        bis = True
        if ('nb' in ll):
            bis = False
        if 'fa' in ll['name']:
            self.layers.add_module(ll['name'], FALinear(in_dim, out_dim, bias=bis, fa=atemp.fa))
        else:
            if 'Lin' in ll:
                scale = 0
                if 'scale' in ll:
                    scale = ll['scale']
                self.layers.add_module(ll['name'], Linear(in_dim, out_dim, scale=scale, iden=False))
            else:
                self.layers.add_module(ll['name'], nn.Linear(in_dim, out_dim, bias=bis))
        if 'zero' in ll:
            temp = getattr(self.layers, ll['name'])
            nn.init.xavier_normal_(temp.weight)
            nn.init.zeros_(temp.bias)

    def process_norm(self, ll, bn, ss):
        if bn == 'full':
            if len(ss) == 4:
                self.layers.add_module(ll['name'], nn.BatchNorm2d(ss[1]))
            else:
                self.layers.add_module(ll['name'], nn.BatchNorm1d(ss[1]))
        elif bn == 'half_full':
            if len(ss) == 4:
                self.layers.add_module(ll['name'], nn.BatchNorm2d(ss[1], affine=False))
            else:
                self.layers.add_module(ll['name'], nn.BatchNorm1d(ss[1], affine=False))
        elif bn == 'layerwise':
            if len(ss) == 2:
                self.layers.add_module(ll['name'], nn.LayerNorm(ss[1]))
            else:
                self.layers.add_module(ll['name'], nn.LayerNorm(ss[1:]))
        elif bn == 'instance':
            self.layers.add_module(ll['name'], nn.InstanceNorm2d(ss[1], affine=True))
        elif bn == 'simple':
            self.layers.add_module(ll['name'], diag2d(ss[1]))
        else:
            self.layers.add_module(ll['name'], Iden())

    def process_conv(self, ll, inp_feats):
        bis = True
        if ('nb' in ll):
            bis = False
        stride = 1;
        if 'stride' in ll:
            stride = ll['stride']
        pd = (ll['filter_size'] // stride) // 2
        if 'fa' in ll['name'] and not 'ga' in pre and 'Darwin' not in os.uname():
            self.layers.add_module(ll['name'],
                                   FAConv2d(inp_feats, ll['num_filters'], ll['filter_size'], stride=stride, fa=atemp.fa,
                                            padding=pd, bias=bis))
            # nn.init.zeros_(p.bias)
        else:
            self.layers.add_module(ll['name'],
                                   nn.Conv2d(inp_feats, ll['num_filters'], ll['filter_size'], stride=stride, padding=pd,
                                             bias=bis))
        if 'zero' in ll:
            temp = getattr(self.layers, ll['name'])
            nn.init.xavier_normal_(temp.weight)
            nn.init.zeros_(temp.bias)

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

                    if everything:
                        OUTS[ll['name']]=out

                if ('shift' in ll['name']):
                     if atemp.first:
                         self.layers.add_module(ll['name'],shifts(ll['shifts']))
                     out=getattr(self.layers,ll['name'])(out)
                     if everything:
                         OUTS[ll['name']] = out

                if ('edge' in ll['name']):
                    if atemp.first:
                        self.layers.add_module(ll['name'],Edge(atemp.dv,slope=atemp.slope))
                        out = getattr(self.layers, ll['name'])(out, torch.device('cpu'))
                    else:
                        out=getattr(self.layers,ll['name'])(out)

                    if everything:
                        OUTS[ll['name']]=out

                if ('conv' in ll['name']):

                    if atemp.first:
                        self.process_conv(ll,inp_feats)


                    if everything:
                        out = OUTS[inp_ind]
                    # Reshape to grid based data with inp_feats features.
                    if len(out.shape)==2:
                        wdim=np.int(np.sqrt(out.shape[1]/inp_feats))
                        out=out.reshape(out.shape[0],inp_feats,wdim,wdim)




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
                        pp = (ll['pool_size'] - 1) // 2
                        if ('stride' in ll):
                            stride = ll['stride']
                            pp=1
                        #pp=[np.int32(np.mod(ll['pool_size'],2))]

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

                if 'nfl' in ll['name']:
                    if atemp.first:
                         self.process_nfl(ll)
                    out,_ = getattr(self.layers, ll['name'])(out)
                    if everything:
                        OUTS[ll['name']] = out

                if ('dense' in ll['name']):
                    if atemp.first:
                        self.process_dense(in_dim,ll)
                    if everything:
                        out=OUTS[inp_ind]
                    if 'Lin' not in ll:
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
                        self.process_norm(ll,atemp.bn,OUTS[old_name].shape)


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
                if ('edge' in ll['name']):
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


def SS_stats(out, OUT, fout):
    with torch.no_grad():
            aa=out[0].reshape(out[0].shape[0],-1).size()
            AA = OUT[0].reshape(out[0].shape[0], -1).size()
            ls=torch.log2(torch.tensor(aa[1],dtype=float))
            LS=torch.log2(torch.min(torch.tensor(AA[0],dtype=float),
                                     torch.tensor(AA[1],dtype=float)))
            _, s, _ = torch.linalg.svd(out[0])
            s = s / torch.sum(s)
            ent = -torch.sum(s * torch.log2(s)) / ls
            _, s, _ = torch.linalg.svd(OUT[0].reshape(out[0].shape[0], -1))
            s = s / torch.sum(s)
            ENT = -torch.sum(s * torch.log2(s)) / LS
            fout.write('\n ent,{:.2F},ENT,{:.4F}\n'.format(ent.cpu().numpy(), ENT.cpu().numpy()))
            l1 = torch.mean(torch.mean(torch.abs(out[0] - out[1]), dim=1))
            s1 = torch.mean(torch.std(torch.abs(out[0] - out[1]), dim=1))
            L1 = torch.mean(torch.mean(torch.abs(OUT[0] - OUT[1]), dim=1))
            S1 = torch.mean(torch.std(torch.abs(OUT[0] - OUT[1]), dim=1))

            fout.write('\n SAME,l1,{:.2F},s1,{:.2F},L1,{:.4F},S1,{:.4F} \n'.format(
                l1.cpu().numpy(), s1.cpu().numpy(), L1.cpu().numpy(), S1.cpu().numpy()))

            out1p = out[1][torch.randperm(out[1].size()[0])]
            OUT1p = OUT[1][torch.randperm(OUT[1].size()[0])]

            l1 = torch.mean(torch.mean(torch.abs(out[1] - out1p), dim=1))
            s1 = torch.mean(torch.std(torch.abs(out[1] - out1p), dim=1))
            L1 = torch.mean(torch.mean(torch.abs(OUT[1] - OUT1p), dim=1))
            S1 = torch.mean(torch.std(torch.abs(OUT[1] - OUT1p), dim=1))

            fout.write('\n DIFFERENT,l1,{:.2F},s1,{:.2F},L1,{:.4F},S1,{:.4F} \n'.format(
                l1.cpu().numpy(), s1.cpu().numpy(), L1.cpu().numpy(), S1.cpu().numpy()))

def run_epoch(model, args, train, epoch, d_type='train', fout='OUT',freq=1):

        if (d_type=='train'):
            model.train()
        else:
            model.eval()

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
        # Dataloader iterator
        # if args.embedd_type=='direct':
        #     if epoch==0:
        #         lossf.alpha=0.
        #     else:
        #         lossf.alpha=args.alpha
        #     print(lossf.alpha)
        tra=iter(train)
        loss_diff=0.
        loss_diff1=0.
        loss_diff2=0.
        loss_diff3=0.
        for j in np.arange(0, num_tr, jump,dtype=np.int32):
            lnum=0
            if d_type=='train':
                optimizer.zero_grad()

            BB, indlist=next(tra)
            # AN image and its augmentation is provided by the data loader
            if type(BB[0]) is list:
                data=[BB[0][0].to(dvv,non_blocking=True),BB[0][1].to(dvv,non_blocking=True)]
            else:
                data_in=BB[0].to(dvv,non_blocking=True)
                data = get_data(data_in, args, dvv, d_type)

            target=BB[1].to(dvv, dtype=torch.long)


            with torch.no_grad() if (d_type != 'train') else dummy_context_mgr():

                out, OUT, data =forw(model,args,data)

                if args.embedd_type is not None and np.mod(epoch,10) == 0 and (j==0 or d_type=='test_stats'):
                    # Some stats of interest to compute in SS learning
                    SS_stats(out, OUT, fout)
                loss, acc = get_loss(lossf,args, out, OUT, target, data)

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
            if args.embedd_type == 'direct':
              with torch.no_grad():
                 # Run the gradient branch through the updated network.
                 outt1=model.forward(data[1])[0]
                 # Everything else stays the same.
                 loss_post = lossf.forw(out[0], outt1)
                 # Non-gradient branch stays the same but covariance is updated.
                 loss_post1 = lossf.forward(out[0],outt1)[0]
                 # Run the non-gradient branch through updated network - this should yield loss like next epoch?
                 outt0 = model.forward(data[0])[0]
                 loss_post2 = lossf.forward(outt0,outt1)[0]

                 loss_diff+=(loss-loss_post).item()
                 loss_diff1+=(loss-loss_post1).item()
                 loss_diff2+=(loss-loss_post2).item()
                 # data = get_data(data_in, args, dvv, d_type)
                 # outt1 = model.forward(data[1])[0]
                 # outt0 = model.forward(data[0])[0]
                 # loss_post3 = lossf.forward(outt0, outt1)[0]
                 # loss_diff3 += (loss - loss_post3).item()
            full_loss[lnum] += loss.item()

            if acc is not None:
                full_acc[lnum] += acc.item()
            count[lnum]+=1
        #if args.embedd_type is not None:
         #   fout.write('\n lossdiff,{:.5F},loss_diff1,{:.5F},loss_diff2,{:.5}\n'.format(loss_diff,loss_diff1,loss_diff2))
        if freq-np.mod(epoch,freq)==1:

           for l in range(ll):
                fout.write('\n ====> Ep {}: {} Full loss: {:.7F}, Full acc: {:.6F} \n'.format(d_type,epoch,
                    full_loss[l] /(count[l]), full_acc[l]/(count[l]*jump)))

        return [full_acc/(count*jump), full_loss/(count)]


def get_data(data_in, args, dvv, d_type):
    if args.embedd_type is not None:

        with torch.no_grad():
            if args.crop == 0:
                data_out = deform_data(data_in, args.perturb, args.transformation, args.s_factor, args.h_factor,
                                       True, dvv)
                if args.double_aug:
                    data_in = deform_data(data_in, args.perturb, args.transformation, args.s_factor, args.h_factor,True, dvv)
                data = [data_in, data_out]
            else:
                data_p = data_in
                data = [data_p[0], data_p[1]]
    else:
        if args.perturb > 0. and d_type == 'train':
            with torch.no_grad():
                data_in = deform_data(data_in, args.perturb, args.transformation, args.s_factor, args.h_factor,
                                      False, dvv)
        data = data_in
    return data

def forw(model, args, input, lnum=0):

    data=input
    OUT0=OUT1=None
    if type(input) is list:

        out1, OOUT1 = model.forward(input[1])
        if args.embedd_type=='AE':
            OUT1=OOUT1['dense_final']
            if args.compare_layers is not None:
                out1=OOUT1[args.compare_layers[1]]
                data1=OOUT1[args.compare_layers[0]]
                print(torch.max(torch.abs(data1)),torch.max(torch.abs(out1)))
        elif args.embedd_type is not None:
            OUT1 = OOUT1[args.embedd_layer]
        with torch.no_grad() if (args.block) else dummy_context_mgr():
            cl = False
            if args.embedd_type == 'clapp':
                cl = True
            out0, OOUT0 = model.forward(input[0], clapp=cl)
            if args.embedd_type == 'AE':
                OUT0 = OOUT0['dense_final']
                if args.compare_layers is not None:
                    out0 = OOUT0[args.compare_layers[1]]
                    data0 = OOUT0[args.compare_layers[0]]
            elif args.embedd_type is not None:
                OUT0 = OOUT0[args.embedd_layer]
        out=[out0,out1]
        OUT=[OUT0,OUT1]
        if args.compare_layers is not None:
            data=[data0,data1]
    else:
        out, OUT = model.forward(input)
        if args.randomize_layers is not None:
            out = OUT[args.randomize_layers[lnum * 2 + 1]]

    return out, OUT, data

def get_loss(aloss, args, out, OUT, target, data=None):

        # Embedding training with image and its deformed counterpart
        if type(out) is list:
            if args.embedd_type == 'AE':
                loss, acc = aloss(out[0], out[1], OUT[0], OUT[1], data[0], data[1])
            else:
                loss, acc = aloss(out[0], out[1])
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
        elif args.sched[0]==-1:
            scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10], gamma=10.)



        return scheduler


