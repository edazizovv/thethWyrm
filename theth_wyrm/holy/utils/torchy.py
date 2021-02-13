
#

import torch
from torch import nn

from scipy import special

"""
class ERF(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        #return torch.from_numpy(special.erf(x.detach().numpy()))
        return special.erf(x.numpy())
"""
"""
import math
from torch.autograd import Function
#class ERF(nn.Module):
class ERF(Function):

#    def __init__(self):
#        super().__init__()

    @staticmethod
    #def forward(ctx, input, hidden):
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.erf(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        return 2. / math.sqrt(math.pi) * torch.exp(-(input ** 2)) * grad_output
"""
import math
from torch.autograd import Function

class ERF_(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.erf(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_variables
        return 2. / math.sqrt(math.pi) * torch.exp(-(input ** 2)) * grad_output


class ERF(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ERF_.apply(input)

"""

class Erf(Function):

    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i.erf()

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_variables
        return 2. / math.sqrt(math.pi) * torch.exp(-(i ** 2)) * grad_output
"""

