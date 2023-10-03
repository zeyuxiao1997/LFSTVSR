# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   train_STLFVSR4_condition2.py
@Time    :   2021/09/22 16:52:43
@Author  :   ZeyuXiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   None
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import random
from models import Losses
from models import LFSTVSR4_condition7
from utils.myutils import *
import torch.backends.cudnn as cudnn
from options import *
from datasets import Dataset4Sintel_condition7
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
from tensorboardX import *
import torchvision.utils as visionutils


def train():
    print(opt)
    Best = 0
    opt.ModelName = 'LFSTVSR4_C7'
    opt.train_batch_size = 6
    opt.upscale_factor = 4
    transform = transforms.Compose([transforms.ToTensor()])
    opt.manualSeed = random.randint(1, 10000)
    opt.saveDir = os.path.join(opt.exp, opt.ModelName)
    # opt.group_file = '/gdata2/zhuyn/VSRBenchmark/TrainingSet/vimeo_septuplet/Train_Vimeo90K_VSR_BDx4.txt'
    create_exp_dir(opt.saveDir)
    opt.patch_size = 64
    # device = torch.device("cuda:7")
    opt.model_path = ''
    opt.group_file = '/disk3/zeyuxData/LFVideo/Sintel_LFV_9x9/LFSTVSRGroups/TrainGroups_Condition7.txt'
    train_data = Dataset4Sintel_condition7.DatasetFromFolder(opt)
    train_dataloader = DataLoader(train_data,
                        batch_size=opt.train_batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)
    print('length of train_dataloader: ',len(train_dataloader))

    # opt.group_file = '/gdata1/liuyt2/LFVideo/Sintel_LFV_9x9/LFSTVSRGroups/LFSTVSRGroupsCondition7/ValidGroups.txt'
    # val_data = Dataset4Sintel_condition7.DatasetFromFolderValid(opt)
    # val_dataloader = DataLoader(val_data,
    #                     batch_size=1,
    #                     shuffle=False,
    #                     num_workers=opt.num_workers,
    #                     drop_last=False)
    # print('length of train_dataloader: ',len(val_dataloader))

    last_epoch = 0

    ## initialize loss writer and logger
    ##############################################################
    loss_dir = os.path.join(opt.saveDir, 'loss')
    loss_writer = SummaryWriter(loss_dir)
    print("loss dir", loss_dir)
    trainLogger = open('%s/train.log' % opt.saveDir, 'w')
    ##############################################################

    ## load teacher network
    ##############################################################
    model = LFSTVSR4_condition7.LFSTVSR()
    model.train()
    model.cuda()

    criterionCharb = Losses.CharbonnierLoss()
    criterionCharb.cuda()

    lr = opt.lr
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        betas=(opt.beta1, opt.beta2))

    iteration = 0
    if opt.model_path != '':
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(opt.model_path, map_location=map_location)
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        iteration = checkpoint["iteration"]
        model.load_state_dict(checkpoint["model"])
        lr = checkpoint["lr"]
        print(lr)
        lr = 1e-5
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('load pretrained')
    
    # model = torch.nn.DataParallel(model)

    AllPSNR = 0
    
    for epoch in range(opt.max_epoch):
        if epoch < last_epoch:
            continue
        for _, (batch) in enumerate(train_dataloader, 0):
            iteration += 1  # 总共的iteration次数

            centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = batch
            centerView1, one1, two1, three1, centerView3, one3, two3, three3, gt = centerView1.cuda(), one1.cuda(), two1.cuda(), three1.cuda(), centerView3.cuda(), one3.cuda(), two3.cuda(), three3.cuda(), gt.cuda()
            # print('centerView1', centerView1.shape)
            # print('one1', one1.shape)
            # print('two1', two1.shape)
            # print('three1', three1.shape)
            # print('centerView3', centerView3.shape)
            # print('one3', one3.shape)
            # print('two3', two3.shape)
            # print('three3', three3.shape)
            # print('gt', gt.shape)

            # # centerView1 torch.Size([4, 3, 64, 64])
            # # one1 torch.Size([4, 8, 3, 64, 64])
            # # two1 torch.Size([4, 8, 3, 64, 64])
            # # three1 torch.Size([4, 8, 3, 64, 64])
            # # centerView3 torch.Size([4, 3, 64, 64])
            # # one3 torch.Size([4, 8, 3, 64, 64])
            # # two3 torch.Size([4, 8, 3, 64, 64])
            # # three3 torch.Size([4, 8, 3, 64, 64])
            # # gt torch.Size([4, 3, 3, 256, 256])

            out = model(centerView1, one1, two1, three1, centerView3, one3, two3, three3)
            # print(out.shape)
            
            optimizer.zero_grad()
            
            CharbLoss = criterionCharb(out, gt)
    #         # SSIMLoss = (1-criterionSSIM(out, gt))/10 #数量级一致
            AllLoss = CharbLoss
            AllLoss.backward()
            optimizer.step()

            prediction = torch.clamp(out,0.0,1.0)

            loss_writer.add_scalar('CharbLoss', CharbLoss.item(), iteration)


            if iteration%2 == 0:
                PPsnr = compute_psnr(tensor2np(prediction[0,1,:,:,:]),tensor2np(gt[0,1,:,:,:]))
                if PPsnr==float('inf'):
                    PPsnr=40
                AllPSNR += PPsnr
                print('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss.item(), PPsnr))
                trainLogger.write(('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss.item(), PPsnr))+'\n')

                # loss_writer.add_scalar('AllLoss', AllLoss.item(), iteration)
                loss_writer.add_scalar('CharbLoss', CharbLoss.item(), iteration)
                # loss_writer.add_scalar('SSIMLoss', SSIMLoss.item(), iteration)
                loss_writer.add_scalar('PSNR', PPsnr, iteration)
                trainLogger.flush()

            if iteration%3000 == 0:
                loss_writer.add_image('Prediction', prediction[0,1,:,:,:], iteration) # x.size= (3, 266, 530) (C*H*W)
                loss_writer.add_image('gt', gt[0,1,:,:,:], iteration)

                
            if iteration % opt.saveStep == 0:
                is_best = AllPSNR > Best
                Best = max(AllPSNR, Best)
                if is_best or iteration%(opt.saveStep*3)==0:
                    prefix = opt.saveDir+'/LFSTVSR_iter{}_'.format(iteration)+'+PSNR_'+str(Best)
                    file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'iteration': iteration,
                        "optimizer": optimizer.state_dict(),
                        "model": model.state_dict(),
                        "lr": lr
                    }
                torch.save(checkpoint, file_name)
                print('model saved to ==>'+file_name)
                AllPSNR = 0

            if (iteration + 1) % opt.decay_step == 0:
                lr = lr * opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    trainLogger.close()




if __name__ == "__main__":
    train()
