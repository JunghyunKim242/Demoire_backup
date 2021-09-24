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
from Util.util_collections import tensor2im, save_single_image,PSNR
from Net.LossNet import L2_LOSS, L1_LOSS, L1_Sobel_Loss
from dataset.dataset import Moire_dataset,AIMMoire_dataset
from torchnet import meter
import colour
import time

def train(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2 3'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if not os.path.exists(args.save_prefix):
        os.makedirs(args.save_prefix)

    print('torch devices = ', args.device)
    print('save_path = ', args.save_prefix)
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

    criterion_l2 = L2_LOSS()
    criterion_l1 = L1_LOSS()
    criterion_sobel_l1 = L1_Sobel_Loss()

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
    list_psnr_output = []
    list_loss_output = []
    list_psnr_input = []
    list_loss_input = []

    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        # scheduler_state = checkpoint["scheduler"]
        # scheduler.load_state_dict(scheduler_state)
        list_psnr_output = checkpoint['list_psnr_output']
        list_loss_output = checkpoint['list_loss_output']
        list_psnr_input = checkpoint['list_psnr_input']
        list_loss_input = checkpoint['list_loss_input']

    for epoch in range(args.max_epoch):
        print('\nepoch = {} / {}'.format(epoch + 1, args.max_epoch))
        start = time.time()
        if epoch < last_epoch:
            continue

        Loss_meter.reset()
        psnr_meter.reset()

        for  ii, (moires, clears, labels) in tqdm(enumerate(train_dataloader)):
            moires = moires.to(args.device)
            clears = clears.to(args.device) #batch,1,256,256(unet,moire)
            outputs = model(moires) # 32,1,256,256 = 32,1,256,256

            # Loss_l1 = criterion_l1(outputs, clears)
            Loss_l2 = criterion_l2(outputs, clears)
            # Loss_sobel_l1 = criterion_sobel_l1(outputs, clears)

            loss = Loss_l2
            optimizer.zero_grad()
            # loss.backward()
            loss.backward(retain_graph = True) # retain_graph = True
            optimizer.step()
            # scheduler.step()

            moires = tensor2im(moires)
            outputs = tensor2im(outputs)
            clears = tensor2im(clears)

            # psnr = colour.utilities.metric_psnr(outputs, clears)
            psnr = PSNR(outputs, clears)
            psnr_meter.add(psnr)
            Loss_meter.add(loss.item())

        model.eval()

        print('training set : \tLoss = {0:0.4f},\tPSNR = {1:0.4f},\t'.format(Loss_meter.value()[0], psnr_meter.value()[0] ))
        loss_output, psnr_output, loss_input, psnr_input = val(model, test_dataloader, epoch,args)
        print('Test set : \tLoss = {0:0.4f},\tPSNR = {1:0.4f},\tinput_PSNR = {2:0.4f} '.format(loss_output, psnr_output, psnr_input))
        list_psnr_output.append( round(psnr_output,4) )
        list_loss_output.append( round(loss_output,4))
        list_psnr_input.append( round(psnr_input,4))
        list_loss_input.append( round(loss_input,4))

        if (epoch + 1) % args.save_every == 0 or epoch == 0:  # 每5个epoch保存一次
            # prefix = args.pthfoler + '{0}_ckpt_epoch{1}_'.format(args.name, epoch + 1)
            # file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            # checkpoint = {  'epoch': epoch + 1,
            #                 "optimizer": optimizer.state_dict(),
            #                 "model": model.state_dict(),
            #                 "lr": lr,
            #                 # "scheduler" : scheduler,
            #                 "list_psnr_output": list_psnr_output,
            #                 "list_loss_output": list_loss_output,
            #                 "list_psnr_input": list_psnr_input,
            #                 "list_loss_input": list_loss_input,
            #                 }
            # torch.save(checkpoint, file_name)

            if psnr_output > args.bestperformance:  # 每5个epoch保存一次
                args.bestperformance = psnr_output
                file_name = args.pthfoler + 'U_Net_Best_{}_ckpt_epoch{:03d}_psnr_{:0.4f}.pth'.format(args.name,epoch+1, round(psnr_output,4) )
                checkpoint = {  'epoch': epoch + 1,
                                "optimizer": optimizer.state_dict(),
                                "model": model.state_dict(),
                                "lr": lr,
                                # "scheduler" : scheduler,
                                "list_psnr_output": list_psnr_output,
                                "list_loss_output": list_loss_output,
                                "list_psnr_input": list_psnr_input,
                                "list_loss_input": list_loss_input,
                                }
                torch.save(checkpoint, file_name)

            with open(args.save_prefix + "PSNR_validation_set_output_psnr.txt", 'w') as f:
                f.write("psnr_output: {:}\n".format(list_psnr_output))
            with open(args.save_prefix + "Loss_validation_set_output_loss.txt", 'w') as f:
                f.write("loss_output: {:}\n".format(list_loss_output))
            with open(args.save_prefix + "PSNR_validation_set_Input_.txt", 'w') as f:
                f.write("input_psnr: {:}\n".format(list_psnr_input))
            with open(args.save_prefix + "Loss_validation_set_Input_.txt", 'w') as f:
                f.write("input_loss: {:}\n".format(list_loss_input))

            if epoch >= 1:
                plt.figure()
                plt.plot(range(1, epoch + 2, 1), list_psnr_output, 'r', label='Validation_set')
                plt.plot(range(1, epoch + 2, 1), list_psnr_input, 'b', label='Input(validation)')
                plt.xlabel('Epochs')
                plt.ylabel('PSNR')
                plt.axis([1, epoch + 1, args.psnr_axis_min, args.psnr_axis_max])  # 10-50
                plt.ylim(10,50)
                plt.title('PSNR per epoch')
                plt.grid(linestyle='--', color='lavender')
                plt.legend(loc='lower right')
                plt.savefig(args.save_prefix + 'PSNR_graph_{name}_{epoch}.png'.format(name=args.name,epoch = epoch+1))
                plt.clf()
        #
        # if psnr_output > args.bestperformance :  # 每5个epoch保存一次
        #     args.bestperformance = psnr_output
        #     prefix = args.pthfoler + 'U_Net_Best_{0}_ckpt_epoch_psnr_{1}'.format(args.name, psnr_output)
        #     file_name = time.strftime(prefix + '.pth')
        #     checkpoint = {
        #         'epoch': epoch + 1,
        #         "optimizer": optimizer.state_dict(),
        #         "model": model.state_dict(),
        #         "lr": lr,
        #         # "scheduler" : scheduler,
        #         "list_psnr_output": list_psnr_output,
        #         "list_loss_output": list_loss_output,
        #         "list_psnr_input": list_psnr_input,
        #         "list_loss_input": list_loss_input,
        #     }
        #     torch.save(checkpoint, file_name)


        if epoch+1 in [50, 100, 200, 300]:#(loss_meter.value()[0] > previous_loss) or ((epoch + 1) % 10) == 0:
            lr = lr *(0.3)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch == (args.max_epoch-1):
            prefix2 = args.pthfoler + '{0}_stdc_epoch{1}.pth'.format(args.name, epoch + 1)
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
        criterion_l1 = L1_LOSS()
        criterion_sobel_l1 = L1_Sobel_Loss()

        loss_output_meter = meter.AverageValueMeter()
        loss_input_meter = meter.AverageValueMeter()
        psnr_output_meter = meter.AverageValueMeter()
        psnr_input_meter = meter.AverageValueMeter()

        image_train_path_demoire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "demoire") if train_sample == False else "{0}/epoch_{1}_training_set_{2}/".format(args.save_prefix, epoch + 1, "demoire")
        if not os.path.exists(image_train_path_demoire) and (epoch + 1) % args.save_every == 0 : os.makedirs(image_train_path_demoire)

        if epoch ==0 :
            image_train_path_moire = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "moire") if train_sample == False else "{0}/epoch_{1}_training_set_{2}/".format(args.save_prefix, epoch + 1, "moire")
            image_train_path_clean = "{0}/epoch_{1}_validation_set_{2}/".format(args.save_prefix, epoch + 1, "clean") if train_sample == False else "{0}/epoch_{1}_training_set_{2}/".format(args.save_prefix, epoch + 1, "clean")
            if not os.path.exists(image_train_path_moire): os.makedirs(image_train_path_moire)
            if not os.path.exists(image_train_path_clean): os.makedirs(image_train_path_clean)

        loss_output_meter.reset()
        loss_input_meter.reset()
        psnr_output_meter.reset()
        psnr_input_meter.reset()

        for ii, (val_moires, val_clears, labels) in tqdm(enumerate(dataloader)):
            val_moires = val_moires.to(args.device)
            val_clears = val_clears.to(args.device)
            val_outputs = model(val_moires)

            # loss_l1 = criterion_l1(val_outputs, val_clears)
            loss_l2 = criterion_l2(val_outputs, val_clears)
            # loss_sobel_l1 = criterion_sobel_l1(val_outputs, val_clears)
            loss = loss_l2
            loss_output_meter.add(loss.item())

            # loss_l1_input = criterion_l1(val_moires, val_clears)
            loss_l2_input = criterion_l2(val_moires, val_clears)
            # loss_sobel_l1_input = criterion_sobel_l1(val_moires, val_clears)
            loss_input = loss_l2_input
            loss_input_meter.add(loss_input.item())

            val_moires = tensor2im(val_moires) # type tensor to numpy .detach().cpu().float().numpy()
            val_outputs = tensor2im(val_outputs)
            val_clears = tensor2im(val_clears)

            bs = val_moires.shape[0]
            if epoch != -1:
                for jj in range(bs):
                    output, clear, moire, label = val_outputs[jj], val_clears[jj], val_moires[jj], labels[jj]

                    # psnr_output_individual = colour.utilities.metric_psnr(output, clear)
                    psnr_output_individual = PSNR(output, clear)
                    # psnr_input_individual = colour.utilities.metric_psnr(moire, clear)
                    psnr_input_individual = PSNR(moire, clear)
                    psnr_output_meter.add(psnr_output_individual)
                    psnr_input_meter.add(psnr_input_individual)

                    if (epoch + 1) % args.save_every == 0 or epoch == 0:  # 每5个epoch保存一次
                        img_path = "{0}/{1}_epoch:{2:04d}_PSNR:{3:.4f}_demoire.png".format(image_train_path_demoire, label,epoch+1 ,psnr_output_individual)
                        save_single_image(output, img_path)

                    if epoch == 0:
                        psnr_in_gt = PSNR(moire, clear)
                        img_path2 = "{0}/{1}_{2:.4f}_moire.png".format( image_train_path_moire, label, psnr_in_gt)
                        img_path3 = "{0}/{1}_clean.png".format(         image_train_path_clean, label)
                        save_single_image(moire, img_path2)
                        save_single_image(clear, img_path3)


        return loss_output_meter.value()[0], psnr_output_meter.value()[0],loss_input_meter.value()[0],psnr_input_meter.value()[0]
