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

if 'Linux' in osu[0] and 'ga' in pre:
    cudnn_convolution = load(name="cudnn_convolution", sources=[datadirs + "cudnn_convolution.cpp"], verbose=True)


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


class NONLIN(nn.Module):
    def __init__(self, ll):
        super(NONLIN, self).__init__()
        self.type=ll['type']

    def forward(self,input):

        if ('HardT' in self.type):
            return(self.HT(input))
        elif ('tanh' in self.type):
            return(F.tanh(input))
        elif ('sigmoid' in self.type):
            return(F.sigmoid(input))
        elif ('relu' in self.type):
            return(F.relu(input))



class Iden(nn.Module):
    def __init__(self):
        super(Iden,self).__init__()

    def forward(self,z):
        return(z)

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





class Linear(nn.Module):
    def __init__(self, dim1,dim2,diag_flag=False, scale=None, iden=False):
        super(Linear, self).__init__()

        # If dimensions are zero just return a dummy variable of the same dimension as input
        self.lin=ident()
        # If diagonal normal with diagonal cov.
        if (diag_flag and dim1>0):
            self.lin=diag(dim1)
        else:
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
            if ctx.device.type=='cpu' or 'ga' not in pre:
                grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
            else:
                grad_weight = cudnn_convolution.convolution_backward_weight(input, weight.shape, grad_output, stride,
                                                                         padding, dilation, groups, False, False)

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


