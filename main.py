#main.py
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from train import train
from test import test
from Net.UNet import UNet
from Net.Mbcnn import MBCNN


####
parser = argparse.ArgumentParser()
parser.add_argument('--traindata_path', type=str,
                    default= '/databse4/jhkim/DataSet/2mura/mura_moire_3_1_variety_135_curve/train2048/', help='vit_patches_size, default is 16')
parser.add_argument('--testdata_path', type=str,
                    default= '/databse4/jhkim/DataSet/2mura/mura_moire_3_1_variety_135_curve/test128/', help='vit_patches_size, default is 16')
parser.add_argument('--testmode_path', type=str,
                    default= '/databse4/jhkim/DataSet/7.capturedmoireimage', help='vit_patches_size, default is 16')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of workers')
parser.add_argument('--batchsize', type=int,default= 16,
                    help='mini batch size')
parser.add_argument('--max_epoch', type=int, default=500,
                    help='number of max_epoch')
parser.add_argument('--lr_decay', type=float,default=0.5,
                    help='adjucting lr_decay ')
parser.add_argument('--accumulation_steps', type=int ,default=1,
                    help='accumulation_steps but not used')
parser.add_argument('--loss_alpha', type=float,default= 0.8,
                    help='weight of charbonnier loss and L2loss, weight for charbonnier loss')
parser.add_argument('--save_every', type=int,default=5,
                    help='saving period for pretrained weight ')
parser.add_argument('--name', type=str,default='U_Net',
                    help='name for this experiment rate')
parser.add_argument('--psnr_axis_min', type=int,default=10,
                    help='mininum line for psnr graph')
parser.add_argument('--psnr_axis_max', type=int,default=70,
                    help='maximum line for psnr graph')
parser.add_argument('--psnrfolder', type=str,default='psnrfoler path was not configured',
                    help='psnrfoler path, define it first!!')
parser.add_argument('--pthfolder', type=str,default='pthfoler path was not configured',
                    help='pthfoler path, define it first!!')
parser.add_argument('--device', type=str, default='cuda or cpu',
                    help='device, define it first!!')
parser.add_argument('--save_prefix', type=str, default='/databse4/jhkim/PTHfolder/210927_U_Net_curvemoire_/',
                    help='saving folder directory')
parser.add_argument('--bestperformance_saveevery', type=float, default=0.,
                    help='saving folder directory')
parser.add_argument('--bestperformance', type=float, default=0.,
                    help='saving folder directory')
parser.add_argument('--pretrained_path', type=str, default = '/databse4/jhkim/DataSet/U_Net_Best_U_Net_ckpt_epoch030_psnr_61.8571_inputpsnr37.3690.pth',
                    help='saving folder directory')
parser.add_argument('--trainmode', type=bool, default=True,
                    help='saving folder directory')

args = parser.parse_args()
if __name__ == "__main__":
    # nFilters=64     multi = True    net = MBCNN(nFilters, multi)
    net = UNet(1, 1)

    # train(args, net)
    test(args, net)


