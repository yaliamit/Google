import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import math,os
import numpy as np
from torch.utils.cpp_extension import load
from data import get_pre
from torch.nn.modules.batchnorm import _NormBase

class _BatchNorm(_NormBase):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.


        if self.momentum is None:
            exponential_average_factor = 0.0
        elif self.training:
            exponential_average_factor = self.momentum
        else:
            exponential_average_factor=0.0
        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)


        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        out= F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)

        return out


class BatchNorm2d(_BatchNorm):


    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))



class BatchNorm1d(_BatchNorm):

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))


pre=get_pre()

datadirs=pre+'Colab Notebooks/STVAE/_CODE/'
osu=os.uname()

#if 'Linux' in osu[0] and not 'ga' in pre:
#    cudnn_convolution = load(name="cudnn_convolution", sources=[datadirs + "cudnn_convolution.cpp"], verbose=True)


class diag2d(nn.Module):
    def __init__(self,dim):
        super(diag2d,self).__init__()
        # dim is number of features
        self.dim = dim
        if (dim>0):
            rim=torch.zeros(dim)
            self.mu=nn.Parameter(rim)
            ris=torch.zeros(dim)
            self.sigma=nn.Parameter(ris)

    def forward(self,z):
        u=z*torch.exp(self.sigma.reshape(1,self.dim,1,1))+self.mu.reshape(1,self.dim,1,1)

        return(u)

class Reshape(nn.Module):
    def __init__(self,sh):
        super(Reshape,self).__init__()

        self.sh=sh

    def forward(self, input):

        out=torch.reshape(input,[-1]+self.sh)

        return(out)

class Inject(nn.Module):
    def __init__(self, ll):
        super(Inject,self).__init__()

        self.ps=ll['stride']
        self.sh=ll['shape']
        self.feats=ll['num_filters']

    def forward(self,input):

        if input.is_cuda:
            num=input.get_device()
            dv=torch.device('cuda:'+str(num))
        else:
            dv=torch.device('cpu')
        input=input.reshape(-1,self.feats, self.sh[0],self.sh[1])
        out=torch.zeros(input.shape[0],input.shape[1],input.shape[2]*self.ps,input.shape[3]*self.ps).to(dv)
        out[:,:,0:out.shape[2]:self.ps,0:out.shape[3]:self.ps]=input

        return out

class NONLIN(nn.Module):
    def __init__(self, type,low=-1., high=1.):
        super(NONLIN, self).__init__()
        self.type=type
        if 'HardT' in self.type:
            self.HT=nn.Hardtanh(low,high)
        if 'smx' in self.type:
            self.tau=high

    def forward(self,input):

        if ('HardT' in self.type):
            return(self.HT(input))
        elif ('tanh' in self.type):
            return(F.tanh(input))
        elif ('sigmoid' in self.type):
            return(torch.sigmoid(input))
        elif ('relu' in self.type):
            return(F.relu(input))
        elif ('smx' in self.type):
            return F.softmax(input*self.tau,dim=1)
        elif ('iden'):
            return(input)

class Channel_Norm(nn.Module):
    def __init__(self,sh):
        super(Channel_Norm,self).__init__()
        self.sh=sh[1:4]
        self.ll=sh[1]*sh[2]
        self.bn=nn.InstanceNorm1d(self.ll)

    def forward(self,input):
        inp=input.reshape(-1,self.sh[1],self.ll).transpose(1,2)
        out=self.bn(inp).transpose(1,2)
        return(out.reshape(-1,self.sh[0],self.sh[1],self.sh[2]))


class Iden(nn.Module):
    def __init__(self):
        super(Iden,self).__init__()

    def forward(self,z):
        return(z)

class Subsample(nn.Module):
    def __init__(self,stride=None):
        super(Subsample,self).__init__()

        self.stride=stride
        if self.stride is not None:
            if stride % 2 ==0:
                self.pd=0
            else:
                self.pd=(stride-1)//2


    def forward(self,z,dv):

        if self.stride is None:
            return(z)
        else:
            if self.pd>0:
                temp=torch.zeros(z.shape[0],z.shape[1],z.shape[2]+2*self.pd,z.shape[3]+2*self.pd).to(dv)
                temp[:,:,self.pd:self.pd+z.shape[2],self.pd:self.pd+z.shape[3]]=z
                tempss=temp[:,:,::self.stride,::self.stride]
            else:
                tempss=z[:,:,::self.stride,::self.stride]


        return(tempss)

class shifts(nn.Module):
    def  __init__(self,shift):
        super(shifts, self).__init__()

        self.shiftx=shift[0]
        self.shifty=shift[1]
        #self.zer=torch.zeros(1,sh[1],sh[2],sh[3]).to(self.dv)

    def forward(self,input,dv):

        temp=torch.zeros(input.shape[0],input.shape[1],
                         input.shape[2]+2*self.shiftx,input.shape[3]+2*self.shifty, device=dv)

        temp[:,:,self.shiftx:self.shiftx+input.shape[2],
                        self.shifty:self.shifty+input.shape[3]]=input
        out=[temp]
        for sx in range(-self.shiftx,self.shiftx+1):
            for sy in range(-self.shifty,self.shifty+1):
                if (sx!=0 or sy!=0):
                    out+=[torch.roll(temp,[sx,sy],dims=[2,3])]

        output=torch.cat(out,dim=1)
        output=output[:,:,self.shiftx:self.shiftx+input.shape[2],
               self.shifty:self.shifty+input.shape[3]]

        return(output)





class biass(nn.Module):
    def __init__(self,dim, scale=None):
        super(biass,self).__init__()

        self.dim=dim
        if (scale is None):
            self.bias=nn.Parameter(6*(torch.rand(self.dim) - .5)/ np.sqrt(self.dim))
        else:
            self.bias=nn.Parameter(scale*(torch.rand(self.dim)-.5))



    def forward(self,z):
        return(self.bias.repeat(z.shape[0],1))

class ident(nn.Module):
    def __init__(self):
        super(ident,self).__init__()

    def forward(self,z):
        return(torch.ones(z.shape[0]))


class diag(nn.Module):
    def __init__(self,dim):
        super(diag,self).__init__()
        self.dim = dim
        if (dim>0):
            rim=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
            self.mu=nn.Parameter(rim)
            ris=(torch.rand(self.dim) - .5) / np.sqrt(self.dim)
            self.sigma=nn.Parameter(ris)

    def forward(self,z):
        u=z*self.sigma+self.mu

        return(u)


class Edge(torch.nn.Module):
    def __init__(self, device, ntr=4, dtr=0):
        super(Edge, self).__init__()
        self.ntr = ntr
        self.dtr = dtr
        self.dv = device
        self.marg=2
        self.delta=3
        self.dirs=[(1,1),(-1,-1),(0,1),(0,-1),(1,0),(-1,0),(-1,1),(1,-1)]
        self.slope=10


    def gt(self,x):

        #y=torch.gt(x,0)
        y=torch.sigmoid(x*self.slope)
        return y

    def forward(self, x):
        x = self.pre_edges(x) #.to(self.dv)
        return x


    def pre_edges(self, im):

        ED=self.get_edges(im)
            # Loop through the 3 channels separately.

        return ED

    def get_edges(self,im):

        sh=im.shape
        delta=self.delta
        pad1=torch.zeros(sh[0],sh[1],sh[2],delta)

        im_a=torch.cat([pad1,im,pad1],dim=3)
        pad2=torch.zeros(sh[0],sh[1],delta,im_a.shape[3])
        im_b=torch.cat([pad2,im_a,pad2],dim=2)





        diff_11 = torch.roll(im_b,(1,1),dims=(2,3))-im_b
        diff_nn11 = torch.roll(im_b, (-1, -1) ,dims=(2,3)) - im_b

        diff_01 = torch.roll(im_b,(0,1), dims=(2,3))-im_b
        diff_n01 = torch.roll(im_b,(0,-1),dims=(2,3))-im_b
        diff_10 = torch.roll(im_b,(1,0), dims=(2,3))-im_b
        diff_n10 = torch.roll(im_b,(-1,0),dims=(2,3))-im_b
        diff_n11 = torch.roll(im_b,(-1,1),dims=(2,3))-im_b
        diff_1n1 = torch.roll(im_b,(1,-1),dims=(2,3))-im_b


        thresh=self.ntr
        dtr=self.dtr
        ad_10=torch.abs(diff_10)
        ad_10=ad_10*self.gt(ad_10-dtr).float()
        e10a=self.gt(ad_10-torch.abs(diff_01)).type(torch.float)\
              + self.gt(ad_10-torch.abs(diff_n01)).type(torch.float) + self.gt(ad_10-torch.abs(diff_n10)).type(torch.float)
        e10b=self.gt(ad_10-torch.abs(torch.roll(diff_01,(1,0),dims=(1,2)))).type(torch.float)+\
                     self.gt(ad_10-torch.abs(torch.roll(diff_n01, (1, 0), dims=(1, 2)))).type(torch.float)+\
                             self.gt(ad_10-torch.abs(torch.roll(diff_01, (1, 0), dims=(1, 2)))).type(torch.float)
        e10 = self.gt(e10a+e10b-thresh) * self.gt(diff_10)
        e10n =self.gt(e10a+e10b-thresh) * self.gt(-diff_10)

        ad_01 = torch.abs(diff_01)
        ad_01 = ad_01*self.gt(ad_10-dtr).float()
        e01a = self.gt(ad_01-torch.abs(diff_10)).type(torch.float) \
               + self.gt(ad_01-torch.abs(diff_n10)).type(torch.float) + self.gt(ad_01-torch.abs(diff_n01)).type(torch.float)
        e01b = self.gt(ad_01-torch.abs(torch.roll(diff_10, (0, 1), dims=(1, 2)))).type(torch.float) + \
                self.gt(ad_01-torch.abs(torch.roll(diff_n10, (0, 1), dims=(1, 2)))).type(torch.float) +\
                    self.gt(ad_01-torch.abs(torch.roll(diff_01, (0, 1), dims=(1, 2)))).type(torch.float)
        e01 = self.gt(e01a + e01b-thresh) * self.gt(diff_01)
        e01n = self.gt(e01a + e01b-thresh) * self.gt(diff_01)



        ad_11 = torch.abs(diff_11)
        ad_11 = ad_11*self.gt(ad_11-dtr).float()
        e11a = self.gt(ad_11-torch.abs(diff_n11)).type(torch.float) \
               + self.gt(ad_11-torch.abs(diff_1n1)).type(torch.float) + self.gt(ad_11-torch.abs(diff_nn11)).type(torch.float)
        e11b = self.gt(ad_11-torch.abs(torch.roll(diff_n11, (1, 1), dims=(1, 2)))).type(torch.float) + \
                self.gt(ad_11-torch.abs(torch.roll(diff_1n1, (1, 1), dims=(1, 2)))).type(torch.float) + \
                    self.gt(ad_11-torch.abs(torch.roll(diff_11, (1, 1), dims=(1, 2)))).type(torch.float)
        e11 = self.gt(e11a + e11b-thresh) * self.gt(diff_11)
        e11n = self.gt(e11a + e11b-thresh) * self.gt(-diff_11)


        ad_n11 = torch.abs(diff_n11)
        ad_n11 = ad_n11 * (ad_n11 > dtr).float()

        en11a= self.gt(ad_n11-torch.abs(diff_11)).type(torch.float) \
               + self.gt(ad_n11-torch.abs(diff_1n1)).type(torch.float) + self.gt(ad_n11-torch.abs(diff_nn11)).type(torch.float)
        en11b = self.gt(ad_n11-torch.abs(torch.roll(diff_11, (-1, 1), dims=(1, 2)))).type(torch.float) + \
               self.gt(ad_n11-torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.float) + \
               self.gt(ad_n11-torch.abs(torch.roll(diff_n11, (-1, 1), dims=(1, 2)))).type(torch.float)
        en11 = self.gt(en11a + en11b-thresh) * self.gt(diff_n11)
        en11=en11.type(torch.float)
        en11n = self.gt(en11a + en11b-thresh) * self.gt(-diff_n11)
        en11n=en11n.type(torch.float)

        marg=self.marg

        edges=torch.zeros(sh[0],sh[1],8,sh[2],sh[3],dtype=torch.float)#.to(self.dv)
        edges[:,:,0,marg:sh[2]-marg,marg:sh[3]-marg]=e10[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,1,marg:sh[2]-marg,marg:sh[3]-marg]=e10n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,2,marg:sh[2]-marg,marg:sh[3]-marg]=e01[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,3,marg:sh[2]-marg,marg:sh[3]-marg]=e01n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,4,marg:sh[2]-marg,marg:sh[3]-marg]=e11[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,5,marg:sh[2]-marg,marg:sh[3]-marg]=e11n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,6,marg:sh[2]-marg,marg:sh[3]-marg]=en11[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]
        edges[:,:,7,marg:sh[2]-marg,marg:sh[3]-marg]=en11n[:,:,delta+marg:delta+sh[2]-marg,delta+marg:delta+sh[3]-marg]


        edges=edges.reshape(-1,8*sh[1],sh[2],sh[3])
        return(edges)


class Linear(nn.Module):
    def __init__(self, dim1,dim2,diag_flag=False, scale=None, iden=False):
        super(Linear, self).__init__()

        # If dimensions are zero just return a dummy variable of the same dimension as input
        self.lin=nn.Identity()
        # If diagonal normal with diagonal cov.
        if (diag_flag and dim1>0):
            self.lin=diag(dim1)
        elif not iden:
            if (dim2>0):
                if (dim1>0):
                    bis = True if dim1>1 else False
                    self.lin=nn.Linear(dim1,dim2,bias=bis)
                    if iden and dim1==dim2:
                        self.lin.weight.data=torch.eye(dim1)
                    if scale is not None:
                        self.lin.bias.data*=scale
                # Only a bias term that does not depend on input.
                else:
                    self.lin=biass(dim2, scale)

    def forward(self,z):
        return self.lin(z)




## Feedback Alignment Linear
class FALinearFunc(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, weight_fb, bias=True, fa=0):
        ctx.save_for_backward(input, weight, weight_fb, bias)
        ctx.fa=fa
        #print('input',input.is_cuda,'weight',weight.is_cuda)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, weight_fb, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_weight_fb = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            if ctx.fa>0:
                grad_input = grad_output.mm(weight_fb) #weight_fb
            else:
                grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if ctx.needs_input_grad[2] and ctx.fa==2:
            grad_weight_fb = grad_weight
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_weight_fb,  grad_bias, None


class FALinear(nn.Module):
    
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True, fa: int= 0) -> None:
        super(FALinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        self.weight_fb = nn.Parameter(torch.Tensor(out_features, in_features))
        self.fa=fa
        #self.weight_fb =torch.Tensor(out_features, in_features) # feedbak weight
        #self.register_buffer('feedback_weight', self.weight_fb)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_fb, a=math.sqrt(5)) # feedback weight
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return FALinearFunc.apply(input, self.weight, self.weight_fb, self.bias, self.fa)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.fa
        )


## Feedback Alignment Conv2d
class FAConv2dFunc(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, weight_fb, bias=True, stride=1, padding=0, dilation=1, groups=1, fa=0, device='cpu'):
        ctx.save_for_backward(input, weight, weight_fb, bias) # Add weight for backward
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.fa=fa
        ctx.device=device
        output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, weight_fb, bias = ctx.saved_tensors # Weight for backward
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups

        grad_input = grad_weight = grad_weight_fb = grad_bias = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]: ## use weight_fb
            #t1 = time.time()
            if ctx.fa>0:
                #if self.dv
                grad_input=torch.nn.grad.conv2d_input(input.shape, weight_fb, grad_output, stride, padding, dilation, groups)
            else:
                grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation,
                                                    groups)
            #print('grad1',time.time()-t1)
        if ctx.needs_input_grad[1]:
             #t3=time.time()
            #if ctx.device.type!='gpu': # or 'ga' not in pre:
            #    grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
            #else:
            grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride,
                                                                         padding, dilation, groups, False, False, False)

             #print('grad2', time.time() - t3)
        if ctx.needs_input_grad[2] and ctx.fa==2:
             grad_weight_fb = grad_weight
        if bias is not None and ctx.needs_input_grad[3]:
            grad_bias = grad_output.sum((0,2,3))

        return grad_input, grad_weight, grad_weight_fb, grad_bias, None, None, None, None, None, None



class FAConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size,  stride=1,
             padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', fa=0, device='cpu'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(FAConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
             False, _pair(0), groups, bias, padding_mode)
        self.weight_fb = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))

        #Initialize
        self.fa=fa
        self.device=device
        nn.init.kaiming_uniform_(self.weight_fb, a=math.sqrt(5))
        
    def forward(self, input):
        if self.padding_mode != 'zeros':
            return FAConv2dFunc.apply(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.weight, self.weight_fb, self.bias, self.stride, _pair(0), self.dilation, self.groups, self.fa, self.device)
        return FAConv2dFunc.apply(input, self.weight, self.weight_fb, self.bias, self.stride,
                        self.padding, self.dilation, self.groups, self.fa, self.device)


