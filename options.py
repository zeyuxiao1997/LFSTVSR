#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   options.py
@Time    :   2020/11/22 18:28:19
@Author  :   ZeyuXiao 
@Version :   1.0
@Contact :   zeyuxiao1997@163.com
@License :   (C)Copyright 2018-2019
@Desc    :   
'''
# here put the import lib

import argparse
import os

parser = argparse.ArgumentParser(description='VSR-Benchmark')

# general settings
parser.add_argument('--ModelName', type=str, default='Test', help='prefix of different dataset')
parser.add_argument('--exp', type=str, default='/disk3/zeyuxData/result', help='prefix of different dataset')
parser.add_argument('--lr', type=float, default=1e-4, help='input batch size')
parser.add_argument('--lr_decay', type=float, default=0.5, help='input batch size')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for ADAM')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
parser.add_argument('--loss_alpha', type=float, default=0.5, help='beta2 for ADAM')
parser.add_argument('--model_path', type=str, default='', help='number of epochs to train for')

parser.add_argument('--BreakCheckpoint', type=str, default='', help='number of epochs to train for')

parser.add_argument('--loss', type=str, default="L1", help="optimizer [Options: SGD, ADAM]")


# dataloader parameters
parser.add_argument('--upscale_factor', type=int, default=4, help="optimizer [Options: SGD, ADAM]")
parser.add_argument('--group_file', type=str, default='', help='file for labels')
parser.add_argument('--augmentation', type=bool, default=True, help='prefix of different dataset')
parser.add_argument('--patch_size', type=int, default=0, help='file for labels')
parser.add_argument('--hflip', type=bool, default=True, help='prefix of different dataset')
parser.add_argument('--rot', type=bool, default=True, help='prefix of different dataset')


# training parameters
parser.add_argument('--train_batch_size', type=int, default=8, help='file for labels')
parser.add_argument('--num_workers', type=int, default=1, help='file for val labels')
parser.add_argument('--max_epoch', type=int, default=8000, help='number of epochs to train for')
parser.add_argument('--decay_step', type=int, default=200000, help='input batch size')
parser.add_argument('--saveStep', type=int, default=3000, help='input batch size')

opt = parser.parse_args()

def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True
