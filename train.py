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
import time


def train(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '3,2,1,0'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('torch devices = ', args.device)
    print('save_path = ', args.save_prefix)


    if not os.path.exists(args.save_prefix):
        os.makedirs(args.save_prefix)
        print(args.save_prefix)
    args.psnrfolder = args.save_prefix + '/psnr_list/'
    args.pthfoler   =  args.save_prefix + '/pth_folder/'
    if not os.path.exists(args.pthfoler)    :   os.makedirs(args.pthfoler)
    if not os.path.exists(args.psnrfolder)  :   os.makedirs(args.psnrfolder)

    Moiredata_train = Moire_dataset(args.traindata_path)
    train_dataloader = DataLoader(Moiredata_train,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)

    Moiredata_test = Moire_dataset(args.testdata_path)
    test_dataloader = DataLoader(Moiredata_test,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)

    model = nn.DataParallel(model)
    model = model.to(args.device)
    model.train()

    L2_loss = L2_LOSS()
    criterion_mse = MSELoss()
    criterion_l1 = L1Loss()
    lr = args.lr
    last_epoch = 0
    optimizer = optim.Adam(params=model.parameters(),
                                 lr=lr,
                                 weight_decay=0.01 #0.005
                                 )

    # scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer,
    #                                         lr_lambda=lambda epoch: 0.95 ** epoch)

    psnr_meter = meter.AverageValueMeter()
    Loss_meter = meter.AverageValueMeter()
    list_val_psnr = []
    list_val_loss = []
    list_val_input_psnr = []
    list_val_input_loss = []


    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]

        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)

        # scheduler_state = checkpoint["scheduler"]
        # scheduler.load_state_dict(scheduler_state)
        list_val_psnr = checkpoint['list_val_psnr']
        list_val_loss = checkpoint['list_val_loss']
        list_val_input_psnr = checkpoint['list_val_input_psnr']
        list_val_input_loss = checkpoint['list_val_input_loss']


    for epoch in range(args.max_epoch):
        print('\nepoch = {} / {}'.format(epoch + 1, args.max_epoch))
        start = time.time()
        if epoch < last_epoch:
            continue

        psnr_meter.reset()

        for  ii, (moires, clears, labels) in tqdm(enumerate(train_dataloader)):
            moires = moires.to(args.device)
            clears = clears.to(args.device) #batch,1,256,256(unet,moire)
            print()
            print('\nIteration = {:04d}\tBefore model(moire)'.format(ii))
            outputs = model(moires) # 32,1,256,256 = 32,1,256,256
            print('After model(moire)')

            # print(outputs.shape) # 32,1,256,256
            # print(outputs.size(0)) # 32

            # print('\ntype of output= \t', type(outputs) )  #
            # print('(output.shape) =\t', outputs.shape )
            # print('type( outputs[0] )= \t',type( outputs[0] ))
            # print(' outputs[0].shape =\t', outputs[0].shape )
            # print('moire.shape =\t',moires.shape)

            # print('clears.shape',clears.shape)
            loss = L2_loss(outputs, clears)
            # loss = criterion_mse(outputs, clears)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()


            moires = tensor2im(moires)
            outputs = tensor2im(outputs)
            clears = tensor2im(clears)

            psnr = colour.utilities.metric_psnr(outputs, clears)
            psnr_meter.add(psnr)
            Loss_meter.add(loss.item())


        model.eval()

        val_loss, val_psnr, val_input_loss, val_input_psnr = val(model, test_dataloader, epoch,args)
        print('Test set : \tLoss = {0:0.4f},\tPSNR = {1:0.4f},\tinput_PSNR = {2:0.4f} '.format(val_loss, val_psnr, val_input_psnr))


        list_val_psnr.append(val_psnr)
        list_val_loss.append(val_loss)
        list_val_input_psnr.append(val_input_psnr)
        list_val_input_loss.append(val_input_loss)


        if (epoch + 1) % args.save_every == 0 or epoch == 0:  # 每5个epoch保存一次
            prefix = args.pthfoler + 'HRnet_{0}_checkpoint_epoch{1}_'.format(args.name, epoch + 1)
            file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            checkpoint = {
                'epoch': epoch + 1,
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict(),
                "lr": lr,
                # "scheduler" : scheduler,
                "list_val_psnr": list_val_psnr,
                "list_val_loss": list_val_loss,
                "list_val_input_psnr": list_val_input_psnr,
                "list_val_input_loss": list_val_input_loss,
            }
            torch.save(checkpoint, file_name)



            with open(args.save_prefix + "PSNR_validation_set_output_{name}_psnr.txt".format(name=args.name), 'w') as f:
                f.write("val_psnr: {:}\n".format(list_val_psnr))
            with open(
                    args.save_prefix + "Loss_validation_set_Charbonnier_output_{name}_loss.txt".format(name=args.name),
                    'w') as f:
                f.write("val_loss: {:}\n".format(list_val_loss))
            with open(args.save_prefix + "PSNR_validation_set_Input_{name}_psnr.txt".format(name=args.name), 'w') as f:
                f.write("input_psnr: {:}\n".format(list_val_input_psnr))
            with open(args.save_prefix + "Loss_validation_set_Input_Charbonnier{name}_loss.txt".format(name=args.name),
                      'w') as f:
                f.write("input_loss: {:}\n".format(list_val_input_loss))

            if epoch >= 1:
                plt.figure()
                plt.plot(range(1, epoch + 2, 1), list_val_psnr, 'r', label='Validation_set')
                plt.plot(range(1, epoch + 2, 1), list_val_input_psnr, 'b', label='Input(validation)')
                plt.xlabel('Epochs')
                plt.ylabel('PSNR')
                plt.axis([1, epoch + 1, args.psnr_axis_min, args.psnr_axis_max])  # 10-50
                plt.ylim(10,max(list_val_psnr))
                plt.title('PSNR per epoch')
                plt.grid(linestyle='--', color='lavender')
                plt.legend(loc='lower right')
                plt.savefig(args.save_prefix + 'PSNR_graph_{name}_{epoch}.png'.format(name=args.name,epoch = epoch+1))
                plt.clf()


        if epoch+1 in [50, 100, 200, 300]:#(loss_meter.value()[0] > previous_loss) or ((epoch + 1) % 10) == 0:
            lr = lr *(0.3)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        if epoch == (args.max_epoch-1):
            prefix2 = args.pthfoler + 'HRnet_{0}_statedict_epoch{1}_'.format(args.name, epoch + 1)
            file_name2 = time.strftime(prefix2 + '%m%d_%H_%M_%S.pth')
            torch.save(model.state_dict(), file_name2)

        print('1 epoch time: {:.2f} seconds \tremaining = {:.2f} minutes, \t {:.2f} hours'.format(
            (time.time() - start),
            ((args.max_epoch - epoch) * (time.time() - start) / 60),
            ((args.max_epoch - epoch) * (time.time() - start) / 3600)))

    return "Training Finished!"


# @torch.no_grad()
with torch.no_grad():
    def val(model, dataloader,epoch,args, vis=None, train_sample = False): # 맨처음 확인할때의 epoch == -1
        model.eval()


        criterion_l2 = L2_LOSS()
        criterion_mse = nn.MSELoss()


        loss_meter = meter.AverageValueMeter()
        SobelLoss_meter = meter.AverageValueMeter()
        L2Loss_meter = meter.AverageValueMeter()
        CLoss_meter = meter.AverageValueMeter()
        loss_input_meter = meter.AverageValueMeter()
        psnr_output_meter = meter.AverageValueMeter()
        psnr_input_meter = meter.AverageValueMeter()
        loss_meter_cri_mse =meter.AverageValueMeter()


        image_train_path_demoire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "demoire") if train_sample == False else "{0}/epoch_{1}_training_set_{2}/".format(args.save_prefix, epoch + 1, "demoire")
        if not os.path.exists(image_train_path_demoire) and (epoch + 1) % args.save_every == 0 : os.makedirs(image_train_path_demoire)


        if epoch ==0 :
            image_train_path_moire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "moire") if train_sample == False else "{0}/epoch_{1}_training_set_{2}/".format(args.save_prefix, epoch + 1, "moire")
            image_train_path_clean = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "clean") if train_sample == False else "{0}/epoch_{1}_training_set_{2}/".format(args.save_prefix, epoch + 1, "clean")
            if not os.path.exists(image_train_path_moire): os.makedirs(image_train_path_moire)
            if not os.path.exists(image_train_path_clean): os.makedirs(image_train_path_clean)


        loss_meter.reset()
        loss_meter_cri_mse.reset()

        for ii, (val_moires, val_clears, labels) in tqdm(enumerate(dataloader)):
            val_moires = val_moires.to(args.device)
            val_clears = val_clears.to(args.device)


            # val_outputs, val_edge_outputs = model(val_moires)
            # val_outputs, _, _ = model(val_moires)

            val_outputs = model(val_moires)

            loss1 = criterion_l2(val_outputs, val_clears)
            loss_meter.add(loss1.item())

            loss_input = criterion_l2(val_moires, val_clears)
            loss_input_meter.add(loss_input.item())

            val_outputs_original = val_outputs
            val_clears_original = val_clears

            val_moires = tensor2im(val_moires) # type tensor to numpy .detach().cpu().float().numpy()
            val_outputs = tensor2im(val_outputs)
            val_clears = tensor2im(val_clears)


            bs = val_moires.shape[0]
            if epoch != -1:
                for jj in range(bs):
                    output, clear, moire = val_outputs[jj], val_clears[jj], val_moires[jj]
                    label = labels[jj]

                    # loss = criterion_l2(output, clear)
                    # loss_meter.add(loss.item())

                    psnr_output_individual = colour.utilities.metric_psnr(output, clear)
                    psnr_input_individual = colour.utilities.metric_psnr(moire, clear)
                    psnr_output_meter.add(psnr_output_individual)
                    psnr_input_meter.add(psnr_input_individual)

                    # if jj ==0:
                        # print('output and clears shape',output.shape, clear.shape)    1,256,256,  1,256,256

                    if (epoch + 1) % args.save_every == 0 or epoch == 0:  # 每5个epoch保存一次
                        img_path = "{0}/{1}_epoch:{2:04d}_PSNR:{3:.4f}_demoire.png".format(image_train_path_demoire, label,epoch+1 ,psnr_output_individual)
                        save_single_image(output, img_path)

                    if epoch == 0:
                        psnr_in_gt = colour.utilities.metric_psnr(moire, clear)
                        img_path2 = "{0}/{1}_{2:.4f}_moire.png".format( image_train_path_moire, label, psnr_in_gt)
                        img_path3 = "{0}/{1}_clean.png".format(         image_train_path_clean, label)
                        save_single_image(moire, img_path2)
                        save_single_image(clear, img_path3)

                    # with open(args.psnrfolder + "psnr_demoire_list_{}.txt".format(label), 'a') as f:
                    #     f.write(",{:0.4f}\t".format(psnr_output_individual))
                    # with open(args.psnrfolder + "loss_Charb_demoire_list_{}.txt".format(label), 'a') as f:
                    #     f.write(",{:0.4f}\t".format(loss_c_indi.item()))

        return loss_meter.value()[0], psnr_output_meter.value()[0],loss_input_meter.value()[0],psnr_input_meter.value()[0]

