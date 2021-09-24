import argparse
import os
import random
import sys
import time
import numpy as np
import torch
from torch import nn
# from torch.nn import MSELoss, L1Loss
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import matplotlib
import matplotlib.pyplot as plt
from Util.util_collections import tensor2im, save_single_image, PSNR
from dataset.dataset import Moire_dataset, AIMMoire_dataset
from torchnet import meter
import colour
import time
from skimage import metrics


def test(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '3,2'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(args.save_prefix):
        os.makedirs(args.save_prefix)

    print('torch devices = ', args.device)
    print('save_path = ', args.save_prefix)

    Moiredata_test = AIMMoire_dataset(args.testmode_path)
    test_dataloader = DataLoader(Moiredata_test,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=False)

    model = nn.DataParallel(model)
    model = model.to(args.device)
    model.eval()

    psnr_output_meter = meter.AverageValueMeter()
    psnr_input_meter = meter.AverageValueMeter()

    image_train_path_moire = "{0}/{1}".format(args.save_prefix, "Moirefolder")
    image_train_path_clean = "{0}/{1}".format(args.save_prefix, "Cleanfolder")
    image_train_path_demoire = "{0}/{1}".format(args.save_prefix, "Demoirefolder")

    if not os.path.exists(image_train_path_moire):      os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean):      os.makedirs(image_train_path_clean)
    if not os.path.exists(image_train_path_demoire):    os.makedirs(image_train_path_demoire)

    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]
        # optimizer_state = checkpoint["optimizer"]
        # optimizer.load_state_dict(optimizer_state)

    for ii ,(moires,clears,labels) in tqdm(enumerate(test_dataloader)):
        moires = moires.to(args.device)
        clears = clears.to(args.device) #batch,1,256,256(unet,moire)
        outputs = model(moires) # 32,1,256,256 = 32,1,256,256

        moires = tensor2im(moires)
        outputs = tensor2im(outputs)
        clears = tensor2im(clears)

        bs = moires.shape[0]
        for jj in range(bs):
            moire, clear, label, output = moires[jj], clears[jj], labels[jj], outputs[jj]

            psnr_output = metrics.peak_signal_noise_ratio(output,clear)
            psnr_output_meter.add(psnr_output)

            psnr_input = metrics.peak_signal_noise_ratio(moire,clear)
            psnr_input_meter.add(psnr_input)

            img_path1 = "{0}/{1}_moire_{2:.4f}.png".format(image_train_path_moire, label, psnr_input)
            save_single_image(moire, img_path1)
            img_path2 = "{0}/{1}_clean.png".format(image_train_path_clean, label)
            save_single_image(clear, img_path2)
            img_path3 = "{0}/{1}_demoire_{2:.4f}.png".format(image_train_path_demoire, label, psnr_output)
            save_single_image(output, img_path3)

    print('TESTdata PSNR = ',psnr_output_meter.value()[0])


