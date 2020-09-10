import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch import Tensor
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import math,os
from torch.utils.cpp_extension import load
from Conv_data import get_pre





pre=get_pre()

datadirs=pre+'Colab Notebooks/STVAE/_CODE/'
#if 'Linux' in os.uname():
#    cudnn_convolution = load(name="cudnn_convolution", sources=[datadirs + "cudnn_convolution.cpp"], verbose=True)


## Feedback Alignment Linear
class FALinearFunc(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, weight_fb, bias=True, fa=0):
        ctx.save_for_backward(input, weight, weight_fb, bias)
        ctx.fa=fa
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
            if ctx.device.type=='cpu':
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


