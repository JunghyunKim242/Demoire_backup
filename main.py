import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# from networks.vit_seg_modeling import VisionTransformer as ViT_seg
# from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
# from config import cfg
# from VAE import VAE
# from VAE_jh import train
from train import train
from test import test
from Net.UNet import UNet
from Net.Mbcnn import MBCNN


parser = argparse.ArgumentParser()

parser.add_argument('--traindata_path', type=str,
                    default= '/databse4/jhkim/DataSet/2mura/mura_moire3_1_lumi135/train800/', help='vit_patches_size, default is 16')
parser.add_argument('--testdata_path', type=str,
                    default= '/databse4/jhkim/DataSet/2mura/mura_moire3_1_lumi135/test100', help='vit_patches_size, default is 16')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num_worker', type=int, default=8,
                    help='number of workers')
parser.add_argument('--batchsize', type=int,default= 1,
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
parser.add_argument('--name', type=str,default='UNet_mura',
                    help='name for this experiment rate')
parser.add_argument('--psnr_axis_min', type=int,default=10,
                    help='mininum line for psnr graph')
parser.add_argument('--psnr_axis_max', type=int,default=50,
                    help='maximum line for psnr graph')
parser.add_argument('--psnrfolder', type=str,default='psnrfoler path was not configured',
                    help='psnrfoler path, define it first!!')
parser.add_argument('--pthfoler', type=str,default='pthfoler path was not configured',
                    help='pthfoler path, define it first!!')
parser.add_argument('--device', type=str, default='cuda or cpu',
                    help='device, define it first!!')
parser.add_argument('--save_prefix', type=str, default='/databse4/jhkim/PTHfolder/210914',
                    help='saving folder directory')
parser.add_argument('--bestperformance', type=int, default=0,
                    help='saving folder directory')
parser.add_argument('--pretrained_path', type=str, default=None,#'/databse4/jhkim/DataSet/2mura/tmpset/HRnet_UNet_mura_checkpoint_epoch500_0910_09_38_16.pth',
                    help='saving folder directory')

parser.add_argument('--trainmode', type=bool, default=True,
                    help='saving folder directory')

args = parser.parse_args()

if __name__ == "__main__":
    # net = VAE()
    # net = UNet(1,1)

    nFilters=64
    multi = True
    print('Before training')
    net = MBCNN(nFilters, multi)
    print('After training')
    train(args, net)
    # test(args, net)


