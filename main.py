import argparse
import urllib
from model import Generator, Discriminator
import torch
from torch import nn

if __name__ == '__main__':
    a = torch.rand((1, 3, 7, 7))
    cnv = torch.nn.Conv2d(3, 10, 1, 3)
    # a = a.view(a.size(0), -1)
    # a = torch.flatten(a, 1)
    print(a.shape)
    # fc = torch.nn.Linear(3*28*28, 19)
    # print(a.shape)
    print(cnv(a).shape)
    # print(fc(a).shape)

