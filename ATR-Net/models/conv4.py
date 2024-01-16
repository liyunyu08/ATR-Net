import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn import init
import functools




def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class BatchNorm2d_fw(nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(x, running_mean, running_var, self.weight.fast, self.bias.fast, training=True,
                               momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(x, running_mean, running_var, self.weight, self.bias, training=True, momentum=1)
        return out

class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(x, self.weight.fast, None, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride=self.stride, padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


# Simple Conv Block
class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out



class ConvNet(nn.Module):
    def __init__(self, depth=4):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i < 3))  # only pooling for fist 4 layers
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)


    def forward(self, x):
        out_0 = self.trunk[0](x)
        out_1 = self.trunk[1](out_0)
        out_2 = self.trunk[2](out_1)
        out_3 = self.trunk[3](out_2)

        return out_3


class ConvNetNopool(
    nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth=4):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool = ( i in [0,1] ), padding = 0 if i in[0,1] else 1 )  # only first two layer has pooling and no padding
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)


    def forward(self, x):
        #   update`
        out_0 = self.trunk[0](x)
        out_1 = self.trunk[1](out_0)
        out_2 = self.trunk[2](out_1)
        out_3 = self.trunk[3](out_2)
        return out_3


class FourLayer_64F(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d, num_classes=5, neighbor_k=3):
        super(FourLayer_64F, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.features = nn.Sequential(  # 3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21
        )


    def forward(self, input1):

        # extract features of input1--query image
        x = self.features(input1)  # torch.Size([50, 64, 21, 21])

        return x


def Conv4():
    return ConvNet(4)


def Conv4Nopool():
    return ConvNetNopool(4)

if __name__ == '__main__':

    spt = torch.randn((5,3,84,84))
    module = Conv4()
    output1,output2 = module(spt)


    print(output1.shape,output2.shape)