import argparse
import os
import random
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss, L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt

from Util.util_collections import tensor2im, save_single_image
from Net.LossNet import L2_LOSS
from dataset.dataset import Moire_dataset
from torchnet import meter
import colour
from skimage import metrics
import time

# @torch.no_grad()
def test(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '3,2,1,0'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('torch devices = ', args.device)
    print('save_path = ', args.save_prefix)


    if not os.path.exists(args.save_prefix):        os.makedirs(args.save_prefix)
    print(args.save_prefix)

    image_testset_path_demoire = "{0}/Testset_Result/".format(args.save_prefix)
    if not os.path.exists(image_testset_path_demoire): os.makedirs(image_testset_path_demoire)

    Moiredata_test = Moire_dataset(args.testdata_path)
    test_dataloader = DataLoader(Moiredata_test,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)

    model = nn.DataParallel(model)
    model = model.to(args.device)
    model.eval()

    L2_loss = L2_LOSS()
    psnr_meter = meter.AverageValueMeter()
    Loss_meter = meter.AverageValueMeter()

    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model'])


    for  ii, (moires, clears, labels) in tqdm(enumerate(test_dataloader)):
        moires = moires.to(args.device)
        clears = clears.to(args.device) #batch,1,256,256(unet,moire)
        outputs = model(moires) # 32,1,256,256 = 32,1,256,256

        loss = L2_loss(outputs, clears)
        Loss_meter.add(loss.item())

        moires = tensor2im(moires)
        outputs = tensor2im(outputs)
        clears = tensor2im(clears)

        bs = moires.shape[0]
        for jj in range(bs):
            output, clear, moire, label = outputs[jj], clears[jj], moires[jj], labels[jj]

            # psnr = colour.utilities.metric_psnr(output, clear)
            psnr = metrics.peak_signal_noise_ratio(output, clear)
            psnr_meter.add(psnr)
            psnr_in_gt = metrics.peak_signal_noise_ratio(moire, clear)

            img_path = "{0}/_Demoire_{1}_PSNR:{2:.4f}_.png".format(image_testset_path_demoire, label, psnr)
            save_single_image(output, img_path)

            img_path2 = "{0}/_Moire_{1}_PSNR:{2:.4f}_.png".format(image_testset_path_demoire, label, psnr_in_gt)
            save_single_image(moire, img_path2)

            img_path3 = "{0}/Clear_{1}_.png".format(image_testset_path_demoire, label)
            save_single_image(clear, img_path3)


    with open(args.save_prefix + "PSNR and Loss Testset.txt", 'w') as f:
        f.write("PSNR: {:}\nLoss{:} = ".format( psnr_meter.value()[0], Loss_meter.value()[0] ))




# with torch.no_grad():
#     def val(model, dataloader,epoch,args, vis=None, train_sample = False): # 맨처음 확인할때의 epoch == -1
#         model.eval()
#         criterion_l2 = L2_LOSS()
#
#         loss_meter = meter.AverageValueMeter()
#         psnr_output_meter = meter.AverageValueMeter()
#
#
#         loss_meter.reset()
#         psnr_output_meter.reset()
#
#         for ii, (val_moires, val_clears, labels) in tqdm(enumerate(dataloader)):
#             val_moires = val_moires.to(args.device)
#             val_clears = val_clears.to(args.device)
#
#
#             val_outputs = model(val_moires)
#
#             loss1 = criterion_l2(val_outputs, val_clears)
#             loss_meter.add(loss1.item())
#
#
#
#             val_moires = tensor2im(val_moires) # type tensor to numpy .detach().cpu().float().numpy()
#             val_outputs = tensor2im(val_outputs)
#             val_clears = tensor2im(val_clears)
#
#
#             bs = val_moires.shape[0]
#                 for jj in range(bs):
#                     output, clear, moire = val_outputs[jj], val_clears[jj], val_moires[jj]
#                     label = labels[jj]
#
#                     psnr_output_individual = colour.utilities.metric_psnr(output, clear)
#                     psnr_output_meter.add(psnr_output_individual)
#
#
#                         img_path = "{0}/{1}_epoch:{2:04d}_PSNR:{3:.4f}_demoire.png".format(image_train_path_demoire, label,epoch+1 ,psnr_output_individual)
#                         save_single_image(output, img_path)
#
#                         psnr_in_gt = colour.utilities.metric_psnr(moire, clear)
#                         img_path2 = "{0}/{1}_{2:.4f}_moire.png".format( image_train_path_moire, label, psnr_in_gt)
#                         img_path3 = "{0}/{1}_clean.png".format(         image_train_path_clean, label)
#                         save_single_image(moire, img_path2)
#                         save_single_image(clear, img_path3)
#
#         return loss_meter.value()[0]
#
