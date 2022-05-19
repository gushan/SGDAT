import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np
from utils import binarize

              

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 784:
            input.data=binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=torch.zeros_like(self.weight)    
        if not hasattr(self.weight,'pre_binary_data'):
            self.weight.pre_binary_data = binarize(self.weight.data)

        self.weight.data=binarize(self.weight)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=torch.zeros_like(self.weight)     
        if not hasattr(self.weight,'pre_binary_data'):
            self.weight.pre_binary_data = binarize(self.weight.data)

        self.weight.data=binarize(self.weight)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
