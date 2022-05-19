import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d



class BinaryNet(nn.Module):

    def __init__(self, input_size=32, num_classes=100):
        super(BinaryNet, self).__init__()
        self.infl_ratio=1
        self.input_size = input_size
        self.features = nn.Sequential(
            nn.Conv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,bias=True),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512 * int(input_size/8) * int(input_size/8), 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            # nn.Dropout(0.5),
            BinarizeLinear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            nn.Linear(1024, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        # self.regime = {
        #     0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
        #     # 40: {'lr': 1e-3},
        #     # 80: {'lr': 5e-4},
        #     # 100: {'lr': 1e-4},
        #     # 120: {'lr': 5e-5},
        #     # 140: {'lr': 1e-5}
        # }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * int(self.input_size/8) * int(self.input_size/8))
        x = self.classifier(x)
        return x


def vgg_cifar10_binary(**kwargs):
    input_size = kwargs.get('input_size', 32)
    num_classes = kwargs.get( 'num_classes', 10)
    return BinaryNet(input_size, num_classes)


def vgg_cifar100_binary(**kwargs):
    input_size = kwargs.get('input_size', 32)
    num_classes = kwargs.get( 'num_classes', 100)
    return BinaryNet(input_size, num_classes)

def vgg_tiny_imagenet_binary(**kwargs):
    input_size = kwargs.get('input_size', 64)
    num_classes = kwargs.get( 'num_classes', 200)
    return BinaryNet(input_size, num_classes)