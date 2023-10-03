#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   my_loader.py
@Time    :   2020/02/08 20:37:11
@Author  :   Zeyu Xiao 
@Version :   1.0
@Contact :   zeyuxiao1997@gmail.com   zeyuxiao1997@163.com
@License :   (C)Copyright 2020-2022, USTC, CHN
@Desc    :   My dataloader
            
'''
# here put the import lib
import os
import os.path
from PIL import Image,ImageOps
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '',
    ]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def get_patch(img_nn, img_tar, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = img_nn[0].size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    # img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = [i.crop((iy, ix, iy + ip, ix + ip)) for i in img_tar]

    return  img_nn, img_tar


def augment(img_nn, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_tar = [ImageOps.flip(i) for i in img_tar]
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_tar = [ImageOps.mirror(i) for i in img_tar]
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_tar = [i.rotate(180) for i in img_tar]
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return  img_nn, img_tar

def get_image(img):
    img = Image.open(img).convert('RGB')
    return img


def load_image_train2(group):
    images = [get_image(img) for img in group]
    targets = images[1:-1]
    inputs = []
    inputs.append(images[0])
    inputs.append(images[-1])
    return inputs, targets


def inputsInterp(inputs):
    im0 = inputs[0]
    im1 = inputs[1]
    im05 = Image.blend(im0, im1, 0.5)
    im025 = Image.blend(im0, im05, 0.5)
    im075 = Image.blend(im05, im1, 0.5)
    interp = []
    interp.append(im0)
    interp.append(im025)
    interp.append(im05)
    interp.append(im075)
    interp.append(im1)
    return interp



class DatasetFromFolder(data.Dataset):
    """
    For test dataset, specify
    `group_file` parameter to target TXT file
    data_augmentation = None
    black_edge_crop = None
    flip = None
    rot = None
    """
    def __init__(self, opt, transform=None):
        super(DatasetFromFolder, self).__init__()
        groups = [line.rstrip() for line in open(os.path.join(opt.group_file))]
        # assert groups[0].startswith('/'), 'Paths from file_list must be absolute paths!'
        self.image_filenames = [group.split('|') for group in groups]
        self.transform = transform
        self.data_augmentation = opt.augmentation
        self.patch_size = opt.patch_size

        if transform is not None:
            self.transforms = transform
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
            ])


    def __getitem__(self, index):

        inputs, targets = load_image_train2(self.image_filenames[index])
        inputs = inputsInterp(inputs)

        if self.patch_size != 0:
            inputs, targets = get_patch(inputs, targets, self.patch_size)
            # inputs, target = inputs, target

        if self.data_augmentation:
            inputs, targets = augment(inputs, targets)

        targets = [self.transforms(i) for i in targets]
        inputs = [self.transforms(j) for j in inputs]

        # inputs = torch.cat((torch.unsqueeze(inputs[0], 0), torch.unsqueeze(inputs[1], 0),
        #                      torch.unsqueeze(inputs[2], 0), torch.unsqueeze(inputs[3], 0),
        #                      torch.unsqueeze(inputs[4], 0)))
        input0 = inputs[0]
        input025 = inputs[1]
        input05 = inputs[2]
        input075 = inputs[3]
        input1 = inputs[4]

        # targets = torch.cat((torch.unsqueeze(targets[0], 0), torch.unsqueeze(targets[1], 0)))

        target025 = targets[0]
        target05 = targets[1]
        target075 = targets[2]

        return input0,input025,input05,input075,input1,target025,target05,target075

    def __len__(self):
        return len(self.image_filenames)




# if __name__ == "__main__":
#     import argparse
#     import os
#     from torch.utils.data import DataLoader


#     parser = argparse.ArgumentParser(description='Temporal-Spatial-SR')
#     parser.add_argument('--decay_step', type=int, default=3e-4, help='input batch size')

#     parser.add_argument('--ModelName', type=str, default='Test', help='prefix of different dataset')
#     parser.add_argument('--exp', type=str, default='/gdata1/xiaozy/ResultAIM2020', help='prefix of different dataset')
#     parser.add_argument('--train_path', type=str, default='/home/aistudio/dataUCF101', help='prefix of different dataset')
#     parser.add_argument('--test_path', type=str, default='/home/aistudio/dataUCF101', help='prefix of different dataset')
#     parser.add_argument('--result_save_path', type=str, default='/gdata/xiaozy/Demoire/TIP2018/testData', help='prefix of different dataset')

#     parser.add_argument('--DataRootHFR', type=str, default='/gdata1/xiaozy/AIM2020/TSRProcessed/TrainsGT', help='prefix of different dataset')
#     parser.add_argument('--DataRootLFR', type=str, default='/gdata1/xiaozy/AIM2020/TSRProcessed/train_15fps', help='prefix of different dataset')

#     parser.add_argument('--trainFileList', type=str, default='/gdata1/xiaozy/AIM2020/TSRProcessed/trainList.txt', help='prefix of different dataset')
#     parser.add_argument('--valFileList', type=str, default='/gdata1/xiaozy/AIM2020/TSRProcessed/valList.txt', help='prefix of different dataset')
#     parser.add_argument('--testFileList', type=str, default='VimeoTrain.txt', help='prefix of different dataset')
#     parser.add_argument('--train_batch_size', type=int, default=1, help='file for labels')

#     parser.add_argument('--is_train', type=bool, default=True, help='prefix of different dataset')
#     parser.add_argument('--up_axis', type=str, default='H', help='prefix of different dataset')
#     parser.add_argument('--DualPath', type=bool, default=False, help='prefix of different dataset')
#     parser.add_argument('--is_Horizontal', type=bool, default=True, help='prefix of different dataset')
#     parser.add_argument('--is_Vertical', type=bool, default=False, help='prefix of different dataset')
#     parser.add_argument('--augmentation', type=bool, default=True, help='prefix of different dataset')

#     parser.add_argument('--in_channels', type=int, default=3, help='file for labels')
#     parser.add_argument('--nf', type=int, default=64, help='file for labels')
#     parser.add_argument('--out_channels', type=int, default=3, help='file for labels')
#     parser.add_argument('--upscale_factor', type=int, default=2, help='file for labels')
#     parser.add_argument('--loss', type=str, default="L2", help="optimizer [Options: SGD, ADAM]")
#     parser.add_argument('--num_workers', type=int, default=4, help='file for val labels')
#     parser.add_argument('--lr', type=float, default=1e-4, help='input batch size')
#     parser.add_argument('--lr_decay', type=float, default=0.9, help='input batch size')
#     parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for ADAM')
#     parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
#     parser.add_argument('--loss_alpha', type=float, default=0.5, help='beta2 for ADAM')
#     # parser.add_argument('--model_path', type=str, default='/home/aistudio/results/Second1e4/Stage2_epoch3+PSNR21065.335395440380311_12_16_02.pth', help='number of epochs to train for')
#     parser.add_argument('--model_path', type=str, default='', help='number of epochs to train for')
#     parser.add_argument('--max_epoch', type=int, default=30, help='number of epochs to train for')
#     parser.add_argument('--modelSR_path', type=str, default='New-SR_epoch3+PSNR99871.290941593590409_04_15_02.pth', help='number of epochs to train for')

#     opt = parser.parse_args()

#     train_data = DatasetFromFolderTrain(opt)
#     train_dataloader = DataLoader(train_data,
#                         batch_size=opt.train_batch_size,
#                         shuffle=True,
#                         num_workers=opt.num_workers,
#                         drop_last=True)
#     print(len(train_dataloader))

#     for iteration, batch in enumerate(train_dataloader, 0):
#         LFR = batch[0]
#         HFR = batch[1]
#         print(LFR.shape,'  ',HFR.shape)


if __name__ == "__main__":
    DataRootLFR = '/gdata1/xiaozy/AIM2020/TSRProcessed/train_15fps'
    DataRootHFR = '/gdata1/xiaozy/AIM2020/TSRProcessed/TrainsGT'
    trainFileList = '/gdata1/xiaozy/AIM2020/TSRProcessed/trainList.txt'    
    VideoFolderList = [line.rstrip() for line in open(trainFileList)]
    VideoFolderPathLFR = [os.path.join(DataRootLFR,x) for x in VideoFolderList]
    VideoFolderPathHFR = [os.path.join(DataRootHFR,x) for x in VideoFolderList]

    VideoNameLFR = VideoFolderPathLFR[0]    # /gdata/xiaozy/ucf101/ApplyEyeMakeup/v_ApplyEyeMakeup_g21_c05
    VideoNameHFR = VideoFolderPathHFR[0]
    

    SequenceLFR = GetVideoFramesLFR(VideoNameLFR)
    SequenceHFR = GetVideoFramesHFR(VideoNameHFR)

    print(SequenceHFR)
    print(SequenceLFR)

    SequenceLFR_TP_THW, SequenceHFR_TP_THW = GetRandomTemProTHW(SequenceLFR, SequenceHFR)
    print(SequenceLFR_TP_THW)
    print(SequenceHFR_TP_THW)

    SequenceLFR_TP_THW.save('lfr.png')
    SequenceHFR_TP_THW.save('hfr.png')
    bibSon = SequenceLFR_TP_THW.resize((181, 720), Image.BICUBIC)
    bibSon.save('biblfr.png')

    SequenceLFR_TP_TWH, SequenceHFR_TP_TWH = GetRandomTemProTWH(SequenceLFR, SequenceHFR)
    print(SequenceLFR_TP_TWH)
    print(SequenceHFR_TP_TWH)

    SequenceLFR_TP_TWH.save('lfr2.png')
    SequenceHFR_TP_TWH.save('hfr2.png')
    bibSon = SequenceLFR_TP_TWH.resize((1280, 181), Image.BICUBIC)
    bibSon.save('biblfr2.png')


    