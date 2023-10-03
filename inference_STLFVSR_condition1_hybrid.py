from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torchvision.transforms import Compose, ToTensor
import numpy as np
import random
from models import Losses
from models import LFSTVSR4
from utils.myutils import *
import torch.backends.cudnn as cudnn
from options import *
from loss import *
# from datasets import Dataset4Sintel_condition1_Baselines
from datasets.Dataset4Sintel_condition1 import load_image
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from tensorboardX import *
import torchvision.utils as visionutils
from myutils.utils import *


def transform():
    return Compose([
        ToTensor(),
    ])

def inference():
    modelname = 'ours4_pure_64'
    model_path = '/disk3/zeyuxData/result/LFSTVSR4_FT_C1/LFSTVSR_iter930000_+PSNR_123954.091382543790105_20_54_42.pth'
    group_file = '/disk3/zeyuxData/LFVideo/hybrid/LFSTVSRGroups/TestGroupsCondition1.txt'
    save_root = '/disk3/zeyuxData/LFVideo/hybrid/Stage2Result/Ours'

    savepath = os.path.join(save_root)
    if not os.path.exists(savepath):                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(savepath)
    groups = [line.rstrip() for line in open(group_file)]
    
    image_filenames = [group.split('|') for group in groups]
    length = len(image_filenames)
    
    compute_lpips_all = perceptual_loss(net='alex')
    compute_ssim_all = ssim_loss()
    compute_psnr_all = psnr_loss()

    model = LFSTVSR4.LFSTVSR()
    model.eval()
    model.cuda()

    if model_path != '':
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(model_path, map_location=map_location)
        iteration = checkpoint["iteration"]
        model.load_state_dict(checkpoint["model"])
        print('load pretrained')


    PSNRall = 0
    SSIMall = 0
    LPIPSall = 0
    iteration = 0
    Timeall = 0

    logger = Logger_yaml(os.path.join(save_root, 'inference_condition1.yml'))

    metric_track = MetricTracker(['ssim', 'psnr', 'lpips'])

    metric_track.reset()
    
    for index in range(0,length,1):
        # print(index)
        iteration = iteration+1
        image1, image2, image3 = image_filenames[index][50:]
        print(image1)
        # print(image1[37:-8])
        ext1 = image1[37:]
        ext2 = image2[37:]
        ext3 = image3[37:]
        folder1 = image1[37:-8]
        folder2 = image2[37:-8]
        folder3 = image3[37:-8]

        centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = load_image(image_filenames[index])
        centerView1 = transform()(centerView1)
        one1 = [transform()(i) for i in one1]
        two1 = [transform()(i) for i in two1]
        three1 = [transform()(i) for i in three1]
        centerView3 = transform()(centerView3)
        one3 = [transform()(i) for i in one3]
        two3 = [transform()(i) for i in two3]
        three3 = [transform()(i) for i in three3]
        gt = [transform()(i) for i in gt]

        one1 = torch.cat((torch.unsqueeze(one1[0], 0), torch.unsqueeze(one1[1], 0),
                             torch.unsqueeze(one1[2], 0), torch.unsqueeze(one1[3], 0),
                             torch.unsqueeze(one1[4], 0), torch.unsqueeze(one1[5], 0),
                             torch.unsqueeze(one1[6], 0), torch.unsqueeze(one1[7], 0)))

        two1 = torch.cat((torch.unsqueeze(two1[0], 0), torch.unsqueeze(two1[1], 0),
                             torch.unsqueeze(two1[2], 0), torch.unsqueeze(two1[3], 0),
                             torch.unsqueeze(two1[4], 0), torch.unsqueeze(two1[5], 0),
                             torch.unsqueeze(two1[6], 0), torch.unsqueeze(two1[7], 0)))

        three1 = torch.cat((torch.unsqueeze(three1[0], 0), torch.unsqueeze(three1[1], 0),
                             torch.unsqueeze(three1[2], 0), torch.unsqueeze(three1[3], 0),
                             torch.unsqueeze(three1[4], 0), torch.unsqueeze(three1[5], 0),
                             torch.unsqueeze(three1[6], 0), torch.unsqueeze(three1[7], 0)))

        one3 = torch.cat((torch.unsqueeze(one3[0], 0), torch.unsqueeze(one3[1], 0),
                             torch.unsqueeze(one3[2], 0), torch.unsqueeze(one3[3], 0),
                             torch.unsqueeze(one3[4], 0), torch.unsqueeze(one3[5], 0),
                             torch.unsqueeze(one3[6], 0), torch.unsqueeze(one3[7], 0)))

        two3 = torch.cat((torch.unsqueeze(two3[0], 0), torch.unsqueeze(two3[1], 0),
                             torch.unsqueeze(two3[2], 0), torch.unsqueeze(two3[3], 0),
                             torch.unsqueeze(two3[4], 0), torch.unsqueeze(two3[5], 0),
                             torch.unsqueeze(two3[6], 0), torch.unsqueeze(two3[7], 0)))

        three3 = torch.cat((torch.unsqueeze(three3[0], 0), torch.unsqueeze(three3[1], 0),
                             torch.unsqueeze(three3[2], 0), torch.unsqueeze(three3[3], 0),
                             torch.unsqueeze(three3[4], 0), torch.unsqueeze(three3[5], 0),
                             torch.unsqueeze(three3[6], 0), torch.unsqueeze(three3[7], 0)))

        gt = torch.cat((torch.unsqueeze(gt[0], 0), torch.unsqueeze(gt[1], 0),
                             torch.unsqueeze(gt[2], 0)))
        

        centerView1 = torch.unsqueeze(centerView1, 0)
        one1 = torch.unsqueeze(one1, 0)
        two1 = torch.unsqueeze(two1, 0)
        three1 = torch.unsqueeze(three1, 0)
        centerView3 = torch.unsqueeze(centerView3, 0)
        one3 = torch.unsqueeze(one3, 0)
        two3 = torch.unsqueeze(two3, 0)
        three3 = torch.unsqueeze(three3, 0)

        # print('centerView1', centerView1.shape)
        # print('one1', one1.shape)
        # print('two1', two1.shape)
        # print('three1', three1.shape)
        # print('centerView3', centerView3.shape)
        # print('one3', one3.shape)
        # print('two3', two3.shape)
        # print('three3', three3.shape)
        # print('gt', gt.shape)
        centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = centerView1.cuda(), one1.cuda(), two1.cuda(), three1.cuda(), centerView3.cuda(), one3.cuda(), two3.cuda(), three3.cuda(), gt.cuda()
        
        with torch.no_grad():
            # lstm_feats = ZoomingSlowMo_part1(inputs)
            # # print(out_ori.shape)
            # # out_ori = torch.clamp(out_ori,0.0,1.0)
            # torch.cuda.empty_cache()

            # outs = ZoomingSlowMo_part2(lstm_feats)
            # # print(out_ori.shape)
            # outs = torch.clamp(outs,0.0,1.0)
            # torch.cuda.empty_cache()
            start = time.time()
            outs = model(centerView1, one1, two1, three1, centerView3, one3, two3, three3)
            end = time.time()
            Timeall += (end-start)
            outs = torch.clamp(outs,0.0,1.0)
            print(outs.shape)

        psnr1 = compute_psnr_all(outs[0,0,:,:,:].cpu(),gt[0,:,:,:]).item()
        psnr2 = compute_psnr_all(outs[0,1,:,:,:].cpu(),gt[1,:,:,:]).item()
        psnr3 = compute_psnr_all(outs[0,2,:,:,:].cpu(),gt[2,:,:,:]).item()
        ssim1 = compute_ssim_all(tensor2np(outs[0,0,:,:,:].cpu()),tensor2np(gt[0,:,:,:])).item()
        ssim2 = compute_ssim_all(tensor2np(outs[0,1,:,:,:].cpu()),tensor2np(gt[1,:,:,:])).item()
        ssim3 = compute_ssim_all(tensor2np(outs[0,2,:,:,:].cpu()),tensor2np(gt[2,:,:,:])).item()
        lpips1 = compute_lpips_all(outs[0,0,:,:,:].cpu(),gt[0,:,:,:]).item()
        lpips2 = compute_lpips_all(outs[0,1,:,:,:].cpu(),gt[1,:,:,:]).item()
        lpips3 = compute_lpips_all(outs[0,2,:,:,:].cpu(),gt[2,:,:,:]).item()
        

        metric_track.update('psnr', psnr1)
        metric_track.update('lpips', lpips1)
        metric_track.update('ssim', ssim1)
        metric_track.update('psnr', psnr2)
        metric_track.update('lpips', lpips2)
        metric_track.update('ssim', ssim2)
        metric_track.update('psnr', psnr3)
        metric_track.update('lpips', lpips3)
        metric_track.update('ssim', ssim3)




        PSNRall = PSNRall+psnr1+psnr2+psnr3
        SSIMall = SSIMall+ssim1+ssim2+ssim3
        LPIPSall = LPIPSall+lpips1+lpips2+lpips3
        print(psnr1, psnr2, psnr3,ssim1, ssim2, ssim3,lpips1, lpips2, lpips3)

        savedir1 = os.path.join(savepath,folder1)
        savedir2 = os.path.join(savepath,folder2)
        savedir3 = os.path.join(savepath,folder3)
        if not os.path.exists(savedir1):
            os.makedirs(savedir1)
        if not os.path.exists(savedir2):
            os.makedirs(savedir2)
        if not os.path.exists(savedir3):
            os.makedirs(savedir3)

        saveimg1 = os.path.join(savepath,ext1)
        saveimg2 = os.path.join(savepath,ext2)
        saveimg3 = os.path.join(savepath,ext3)

        img1 = ToPILImage()(outs[0,0,:,:,:].cpu())
        img2 = ToPILImage()(outs[0,1,:,:,:].cpu())
        img3 = ToPILImage()(outs[0,2,:,:,:].cpu())
        # img1.save(saveimg1)
        # img2.save(saveimg2)
        # img3.save(saveimg3)

        if not os.path.exists(saveimg1):
            img1.save(saveimg1)
            print('saveend1',saveimg1)

        if not os.path.exists(saveimg2):
            img2.save(saveimg2)
            print('saveend2',saveimg2)

        if not os.path.exists(saveimg3):
            img3.save(saveimg3)
            print('saveend3',saveimg3)

    print(PSNRall/(iteration*3))    
    print(SSIMall/(iteration*3))    
    print(LPIPSall/(iteration*3))  
    print(Timeall/(iteration*3))
    result = metric_track.result()
    all_data = metric_track.all_data()
    logger.log_dict(result, 'evaluation results')
    logger.log_dict(all_data, 'all data')
    

if __name__ == "__main__":
    inference()
