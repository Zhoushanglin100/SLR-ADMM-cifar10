import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


# binarize multiply mean value
# def Binarize(tensor,quant_mode='determined'):
# if quant_mode=='determined':
# n = tensor[0].nelement() #number of element
# s = tensor.size()
# if len(s) == 4: #convolution layer
# m = tensor.norm(1, 3, keepdim=True)\
# .sum(2, keepdim=True).sum(1, keepdim=True).div(n) # alpha = sum absolute values / c*w*h
# elif len(s) == 2: # fully connected layer
# m = tensor.norm(1, 1, keepdim=True).div(n) #alpha = sum absolute values/c*w*h
# tensor = tensor.sign().mul(m.expand(s))
# return tensor
# else:
# return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)


def ternarize(tensor, quant_mode='scale'):

    if quant_mode == 'scale':

        alpha = torch.mean(torch.abs(tensor))

    return alpha*(tensor.sign())


# binarize using sign function
def binarize(tensor, quant_mode='determined'):
    if quant_mode == 'determined':
        return tensor.sign()
    if quant_mode == 'scale':
        alpha = torch.mean(torch.abs(tensor))
        # Q = np.zeros(tensor.shape)
        Q= np.where(tensor/alpha >= 0.5, 1, 0)
        alpha = np.sum(np.multiply(tensor, Q))
        QtQ = np.sum(Q ** 2)
        alpha /= QtQ
        return tensor
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0, 1).round().mul_(2).add_(-1)


def ternarize_weight(tensor, quant_mode='determined'):
    if quant_mode == 'determined':
        n = tensor[0].nelement()
        s = tensor.size()
        if len(s) == 4:
            m = tensor.norm(1, 3).sum(2).sum(1).div(n)
            m = m[0].item()
        if len(s) == 2:
            m = tensor.norm(1, 1).div(n)
            m = m[0].item()
        temp_pos = (tensor > (0.7 * m)).type(torch.FloatTensor)
        temp_neg = (tensor < (-0.7 * m)).type(torch.FloatTensor)
        temp = temp_pos - temp_neg
    elif quant_mode=='scale':
        temp=None
    return temp


# Ternarize with mean value
# def Ternarize_weight(tensor):
# n = tensor[0].nelement()
# s = tensor.size()
# if len(s) == 4:
# m_t = tensor.norm(1,3,keepdim=True)\
# .sum(2, keepdim=True).sum(1,keepdim=True).div(n)
# m = tensor.norm(1,3).sum(2).sum(1).div(n)
# m = m[0].item()
# if len(s) == 2:
# m_t = tensor.norm(1,1,keepdim=True).div(n)
# m = tensor.norm(1,1).div(n)
# m = m[0].item()
# temp_pos = (tensor>(0.7*m)).type(torch.FloatTensor)
# temp_neg = (tensor<(-0.7*m)).type(torch.FloatTensor)
# temp = temp_pos-temp_neg
# temp = temp.mul(m_t.expand(s))

# return temp

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self, input, target):
        # import pdb; pdb.set_trace()
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input, target)


class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction, self).__init__()
        self.margin = 1.0

    def forward(self, input, target):
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        self.save_for_backward(input, target)
        loss = output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self, grad_output):
        input, target = self.saved_tensors
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        import pdb;
        pdb.set_trace()
        grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(input.numel())
        return grad_output, grad_output


def Quantize(tensor, quant_mode='det', params=None, numBits=8):
    tensor.clamp_(-2 ** (numBits - 1), 2 ** (numBits - 1))
    if quant_mode == 'det':
        tensor = tensor.mul(2 ** (numBits - 1)).round().div(2 ** (numBits - 1))
    else:
        tensor = tensor.mul(2 ** (numBits - 1)).round().add(torch.rand(tensor.size()).add(-0.5)).div(2 ** (numBits - 1))
    return tensor


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        # if input.size(1) != 784:
        input.data = binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class TernarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        # if input.size(1) != 784:
        input.data = binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = ternarize_weight(self.weight.org).cuda()
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


# class BinarizeConv2d(nn.Module): # change the name of BinConv2d
#    def __init__(self, input_channels, output_channels,
#            kernel_size=-1, stride=-1, padding=-1, groups=1, dropout=0, size=0):
#        super(BinarizeConv2d, self).__init__()
#        self.input_channels = input_channels
#        self.layer_type = 'BinarizeConv2d'
#        self.kernel_size = kernel_size
#        self.stride = stride
#        self.padding = padding
#        self.biconv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, 
#                      stride=stride, padding=padding, groups=groups)
#    
#    def forward(self, x):
#        x = Binarize(x)
#        if not hasattr(self.weight, 'org'):
#            self.weight.org=self.weight.data.clone()
#        self.weight.data=Binarize(self.weight.org)
#        
#        out = self.biconv(x)
#        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        # if input.size(1) != 2:
        input.data = binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


class TernarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        # if input.size(1) != 2:
        input.data = binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = ternarize_weight(self.weight.org).cuda()

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
