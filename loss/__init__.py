import torch
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity
from skimage.measure import compare_ssim as ssim

import sys  
sys.path.append('loss.PerceptualSimilarity.models')  
import  loss.PerceptualSimilarity as models


class L1Loss():
    def __init__(self):
        super().__init__()

        self.L1 = nn.L1Loss()

    def __call__(self, list1, list2):
        loss = 0
        for item1, item2 in zip(list1, list2):
            loss += self.L1(item1, item2)

        return loss


class perceptual_loss():
    def __init__(self, weight=1.0, net='alex', use_gpu=True, gpu_ids=[0]):
        """
        Wrapper for PerceptualSimilarity.models.PerceptualLoss
        """
        self.model = models.PerceptualLoss(net=net, use_gpu=use_gpu, gpu_ids=gpu_ids)
        self.weight = weight

    def __call__(self, pred, target, normalize=True):
        """
        pred and target are Tensors with shape N x C x H x W (C {1, 3})
        normalize scales images from [0, 1] to [-1, 1] (default: True)
        PerceptualLoss expects N x 3 x H x W.
        """
        if pred.shape[1] == 1:
            pred = torch.cat([pred, pred, pred], dim=1)
        if target.shape[1] == 1:
            target = torch.cat([target, target, target], dim=1)
        dist = self.model.forward(pred, target, normalize=normalize)
        return self.weight * dist.mean()


class ssim_loss():
    def __init__(self):
        self.ssim = SSIM

    def __call__(self, im1, im2):
        isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
        s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
        return s


class psnr_loss():
    def __init__(self):
        self.psnr = PSNR

    def __call__(self, pred, tgt):
        loss = self.psnr(tgt.squeeze().cpu().numpy(), pred.squeeze().cpu().numpy())

        return loss