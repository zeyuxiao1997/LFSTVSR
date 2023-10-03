from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchvision import models

from torchvision import models
import torchvision

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load('/home/aistudio/data/data27299/vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return h_relu4
        # return h_relu1


class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # loss = 0
        # for i in range(len(x_vgg)):
        loss =  self.criterion(x_vgg, y_vgg.detach())
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


def mse_loss(pred, target):
    """L2 loss or MSE loss"""
    return F.mse_loss(pred, target, reduction='none')

class AbsoluteSumTS(nn.Module):
    """Sum of absolute values"""

    def __init__(self):
        super(AbsoluteSumTS, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, featureT, featureS):
        B,C,H,W = featureT.shape
        featureT = torch.abs(Variable(featureT))
        featureS = torch.abs(Variable(featureS))
        featureT = torch.sum(featureT,dim=1)
        featureS = torch.sum(featureS,dim=1)
        # featureT = F.softmax(featureT,dim=None)
        # featureS = F.softmax(featureS,dim=None)
        return self.loss(featureS, featureT)


class AbsoluteSumSquareTS(nn.Module):
    """Sum of the square value of absolute values"""
    def __init__(self):
        super(AbsoluteSumSquareTS, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, featureT, featureS):
        B,C,H,W = featureT.shape
        featureT = torch.abs(featureT)
        featureS = torch.abs(featureS)
        featureT = featureT*featureT  # 平方
        featureS = featureS*featureT  # 平方
        featureT = torch.sum(featureT,dim=1)
        featureS = torch.sum(featureS,dim=1)
        # featureT = F.softmax(featureT,dim=None)
        # featureS = F.softmax(featureS,dim=None)
        # featureT = torch.where(torch.isnan(featureT), torch.full_like(featureT, 1), featureT)
        # featureS = torch.where(torch.isnan(featureS), torch.full_like(featureS, 1), featureS)
        # featureT = torch.where(torch.isinf(featureT), torch.full_like(featureT, 1), featureT)
        # featureS = torch.where(torch.isinf(featureS), torch.full_like(featureS, 1), featureS)
        return self.loss(featureS, featureT)

class AbsoluteSumMaxTS(nn.Module):
    """Sum of the square value of absolute values"""
    def __init__(self):
        super(AbsoluteSumMaxTS, self).__init__()
        self.loss = nn.MSELoss()

    def forward(self, featureT, featureS):
        B,C,H,W = featureT.shape
        featureT = torch.abs(featureT)
        featureS = torch.abs(featureS)
        featureT = featureT*featureT  # 平方
        featureS = featureS*featureS  # 平方
        featureT = torch.max(featureT,dim=1)[0]
        featureS = torch.max(featureS,dim=1)[0]
        # featureT = F.softmax(featureT,dim=None)
        # featureS = F.softmax(featureS,dim=None)
        return self.loss(featureS, featureT)


if __name__ == "__main__":
    images = Variable(torch.ones(1, 3, 128, 128)).cuda()
    vgg = Vgg19()
    vgg.cuda()
    print("do forward...")
    outputs = vgg(images)
    print (outputs.size())   # (10, 100)
    print(torch.max(outputs))
