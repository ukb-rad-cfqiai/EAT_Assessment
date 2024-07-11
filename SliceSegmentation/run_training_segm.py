#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2024 Clinic for Diagnositic and Interventional Radiology, University Hospital Bonn, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import os
parent_directory = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
sys.path.append(parent_directory) 
import argparse
from tqdm import tqdm
import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter
import pickle
from torchvision.utils import make_grid
import pandas as pd
import nibabel as nib
import logging
from utils.DataOrganizerSegmentation import organizeData
from accelerate import Accelerator
from utils.Metrics import diceCoef
import shutil
from infer.SliceSegmentation import SliceSegmentation
import datetime
from skimage import color

from monai.networks.nets import UNet
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import GridSamplePadMode, set_determinism
from monai.transforms import (
    Compose,
    LoadImaged,
    RandAffined,
    RandAxisFlipd,
    RandGaussianNoised,
    ScaleIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
)
from monai.losses import DiceLoss
from monai.losses.dice import *

def save_obj_pkl(path, obj):
    with open( path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj_pkl(path ):
    with open( path, 'rb') as f:
        return pickle.load(f)

def overlaySegmentation(image, label, num_plot=5):
    image_plot = image[0:num_plot,:].squeeze()
    image_plot = torch.flip(image_plot,dims=[1]).cpu().detach().numpy()
    label_plot = label[0:num_plot,:].squeeze()
    label_plot = torch.flip(label_plot,dims=[1]).cpu().detach().numpy()
    image_plot = (image_plot - np.min(image_plot)) * 255 / (np.max(image_plot)-np.min(image_plot))
    image_plot = image_plot.astype(np.uint8)

    result_image = torch.tensor(color.label2rgb(label_plot, image_plot))
    result_image = result_image.permute((0,3,1,2))
    return result_image 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to input data", type=str, required=True)
    parser.add_argument("--output_path", help="path, where to save output data", type=str, required=True)
    parser.add_argument("-bs","--batch_size", help="set batch size for training (default:200)", default=200, type=int)
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",required=False, action="store_true")
    parser.add_argument("--num_workers", help="num workers used for data loading, default: 4", default=4, type=int)
    parser.add_argument("-e","--epochs", help='number of epochs for training, default:10000', default=10000,type=int)
    parser.add_argument("--max_steps", help='Max Steps for optimizer. Set to -1 for total steps', default=-1,type=int)
    parser.add_argument("--lr", help='initial learning rate for training, default: 1e-3', default=1e-3, type=float)
    parser.add_argument("--wd", help='weight decay, default: 0.1', default=0.1, type=float)
    parser.add_argument("-sf", "--save_freq", help='save every sf epoch', default=100, type=int)
    parser.add_argument("--seed", help='seed', default=42, type=int)
    parser.add_argument("--fp16", help='Use this for mixed precision training',required=False, action="store_true")
    parser.add_argument("--use_cache", help='Use cache',required=False, action="store_true")
    parser.add_argument("--pin_memory", help='Pin memory on gpu',required=False, action="store_true")
    parser.add_argument('--imsize','--list', nargs='+', default=None,
                        help='Image size. dafault gets im size from first training image') 
    parser.add_argument('--augm-prob-flip', type=float, default=0.1, help='probability to apply flip augmentation. default 0.1') 
    parser.add_argument('--augm-prob-affine', type=float, default=0.75, help='probability to apply affine augmentation. default 0.75')          
    parser.add_argument('--augm-shear-range', type=float, default=0.1, help='shear range for augmemntation. default 0.1') 
    parser.add_argument('--augm-rot-range', type=float, default=20, help='Rotation range for augmentation. defualt 20')      
    parser.add_argument('--augm-scale-range', type=float, default=0.1,  help='Sclae range for augmentation. default 0.1')  
    parser.add_argument('--augm-prob-noise', type=float, default=0.5, help='probability to apply gaussian noise augmentation. default 0.5')       
    parser.add_argument('--augm-std-noise', type=float, default=0.01, help='Std of gaussian noise. default 0.01')  
    parser.add_argument("--reduceLROnPlateau", help='Use this if you want to use ReduceLROnPlateu, else: OneCycleLR',required=False, action="store_true")
    parser.add_argument("--earlyStopping", help='Use this if you want to perform early stopping',required=False, action="store_true")
    parser.add_argument("--earlyStoppingPatience", help="Define after which number of epochs without improvement early stopping should be performed, default 100", type=int, default=100)
    parser.add_argument("--loss", help="Specify which loss you want to use for train (ce_dice, ce or dice)", type=str, default='ce_dice')
    parser.add_argument("--gradAccum", help='number of gradient accumulations, default:2', default=2, type=int)
    parser.add_argument("--gpuNum", help='set number of gpu you want to use', default=0, type=int)
    parser.add_argument("--trainPercentage", help="Specify ratio of training data, default=0.7", default=0.7, type=float)
    parser.add_argument("--valPercentage", help="Specify ratio of validation data, default=0.15", default=0.15, type=float)
    parser.add_argument("--split_path", help="If you do not want random split, specify path to .csv table, where split into train/val/test is defined", type=str, default=None)
    parser.add_argument("--gtSegmIdentifier", help="Specify file ending of ground truth segmentations (e.g. segm.nii.gz, segm.nii or segm.seg.nrrd). Default: segm.nii.gz", type=str, required=False, default='segm.nii.gz')
    parser.add_argument("--slicerSegmName", help="Specify segmetation name for segmentations from 3D Slicer. Only relevant if gtSegmIdentifier == .seg.nrrd", type=str, required=False, default=None)
    parser.add_argument("--plotResults", help="Use this, if you want to visualize data, e.g. segmentations", action="store_true")
    parser.add_argument("--evalTest", help="Use this, if you want to predict test data after completed training", action="store_true")
    parser.add_argument("--evalVal", help="Use this, if you want to predict validation data after completed training", action="store_true")

    args = parser.parse_args()
    
    
    now = datetime.datetime.now()
    model_path = os.path.join(args.output_path,'trained_models',f'model_{now.strftime("%Y-%m-%d_%H-%M-%S")}')
    
    if not os.path.exists(model_path): os.makedirs(model_path) 
    
    logging.root.handlers = []
    handlers=[logging.StreamHandler(sys.stdout)]
    handlers.append( logging.FileHandler(os.path.join(model_path,'train.log')))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers)

    logging.info(args)
    
    if args.seed is not None:
        logging.info(f'Applying seed {args.seed}')
        set_determinism(seed=args.seed)
        
    data = []
    for root_path, dirs, files in os.walk(args.data_path):
        for name in files:
            if (name.endswith('.nii.gz') or name.endswith('.nii')) and not name.startswith('.') and not args.gtSegmIdentifier in name:
                data.append(os.path.join(root_path,name))
    data.sort()
    
    segm = []
    for root_path, dirs, files in os.walk(args.data_path):
        for name in files:
            if args.gtSegmIdentifier in name and not name.startswith('.'):
                segm.append(os.path.join(root_path,name))
    segm.sort()    
     
    print('-----------------------Organize data for training -----------------')
    data_df, df_orgAndAIPath = organizeData(data, segm, args.output_path, args.trainPercentage, args.valPercentage, 
                                       args.split_path, args.slicerSegmName, args.plotResults)
    
    data_df.to_csv(os.path.join(args.output_path,f'dataset_{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'))
    df_orgAndAIPath.to_csv(os.path.join(args.output_path,f'orgAndAIPath_{now.strftime("%Y-%m-%d_%H-%M-%S")}.csv'))
    
    print('--------------------- Start training ------------------------------')
    data_dicts = np.asarray([
        {'img': img, 'segm': segm}
        for img, segm in 
        zip(data_df['img'].to_numpy(),data_df['segm'].to_numpy())])
    
    isVal = data_df['isVal'].astype(bool)
    isTest = data_df['isTest'].astype(bool)
    
    data_dict_train = data_dicts[np.logical_and(~isTest, ~isVal)]
    data_dict_val = data_dicts[isVal]
    logging.info(f'train cases {len(data_dict_train)} ( {np.round(len(data_dict_train)/len(data_dicts) *100,2)}%)')
    logging.info(f'valid cases {sum(isVal)} ( {np.round(sum(isVal)/len(data_dicts) *100,2)}%)')

    if args.evalTest:
        data_dict_test = data_dicts[isTest]
        logging.info(f'test  cases {sum(isTest)} ( {np.round(sum(isTest)/len(data_dicts) *100,2)}%)') 
    
    if args.imsize is None:
        example = nib.load(data_dict_train[0]['img']).get_fdata()
        args.imsize = example.shape    
        example = None
    
    train_transforms = Compose([
        LoadImaged(keys=['img', 'segm']),
        EnsureTyped(keys=['img','segm'], data_type='tensor'),
        EnsureChannelFirstd(keys=['img','segm']),
        RandAxisFlipd(
            keys = ['img','segm'],
            prob = args.augm_prob_flip),
        RandAffined(
            keys = ['img','segm'],
            mode = ('bilinear', 'nearest'),
            prob = args.augm_prob_affine, 
            shear_range = args.augm_shear_range,
            spatial_size = args.imsize,
            rotate_range = np.pi/180*args.augm_rot_range,
            scale_range = args.augm_scale_range,
            padding_mode = GridSamplePadMode.BORDER), 
        RandGaussianNoised(
            keys = ['img'],
            prob = args.augm_prob_noise, 
            std = args.augm_std_noise,
            ),
        ScaleIntensityd(keys=['img'],minv=0,maxv=1)
        ])
    
    eval_transforms = Compose([
        LoadImaged(keys=['img','segm']),
        EnsureTyped(keys=['img','segm'], data_type='tensor'),
        EnsureChannelFirstd(keys=['img','segm']),
        ScaleIntensityd(keys=['img'],minv=0,maxv=1)
        ])

    if args.use_cache:
        train_ds = CacheDataset(
            data=data_dict_train, transform=train_transforms,
            cache_rate=1.0, num_workers=args.num_workers)
        val_ds = CacheDataset(
            data=data_dict_val, transform=eval_transforms,
            cache_rate=1.0, num_workers=args.num_workers)
    else:
        train_ds = Dataset(data=data_dict_train, transform=train_transforms)
        val_ds = Dataset(data=data_dict_val, transform=eval_transforms)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=args.pin_memory, drop_last=True)
    
    valid_loader = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=args.pin_memory)
    
    
    latest_model_path = '%s/latest_model.model' %model_path
    best_model_path = '%s/best_model.model' %model_path
    final_model_path = '%s/final_model.model' %model_path
    
    if args.gpuNum > torch.cuda.device_count():
        args.gpuNum = torch.cuda.device_count()-1
    device = torch.device(f'cuda:{args.gpuNum}' if torch.cuda.is_available() else 'cpu')
    
    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2, 
        channels=(64, 128, 256, 512, 1024),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )     
  
    optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.wd)

    start_epoch = 1
    if args.continue_training:
        try:
            checkpoint = torch.load(latest_model_path)
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
        except:
            print('Warning: load latest checkpoint failed for %s' %latest_model_path)
            start_epoch = 1

    # FOR TENSORBOARD
    writer = SummaryWriter(model_path+os.sep+'runs')
    
    writer_dict = { 'epochs': [],
                    'lr': [], 
                    'loss_train': [],
                    'loss_valid': [],
                    'dice_train':[],
                    'dice_valid':[],
                    'walltime': []
                    }
    
    logging_dict = {'LR': args.lr,
                    'WD': args.wd,
                    'loss':args.loss,
                    'BatchSize': args.batch_size,
                    'Epochs': args.epochs,
                    'reduceLROnPlateau': args.reduceLROnPlateau,
                    'GradAccum': args.gradAccum,
                    'EarlyStopping': args.earlyStopping,
                    }
    
    if args.earlyStopping:
        logging_dict['EarlyStoppingPatience'] = args.earlyStoppingPatience
    
    loss_dice = DiceLoss(include_background=False,softmax=True).to(device) 
    loss_ce = torch.nn.CrossEntropyLoss().to(device)

    accum_iter = args.gradAccum
    num_steps = int((args.epochs*len(train_loader))/accum_iter) if args.max_steps<0 else args.max_steps #for Debug !
    
    if not args.reduceLROnPlateau:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=args.lr,
                                                        total_steps=num_steps) 
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,min_lr = 1e-6, patience=30, threshold=1e-5) 

    if args.fp16:
        accelerator = Accelerator(gradient_accumulation_steps=accum_iter,mixed_precision='fp16')
    else:
        accelerator = Accelerator(gradient_accumulation_steps=accum_iter)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    model.to(device)

    min_valid_loss = None
    
    steps = 0
    count_earlyStopping = 0

    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        start_time = time.time()
        
        writer_dict['epochs'].append(epoch)
        writer.add_scalar('utils/epochs', epoch, steps)

        model.train()
        train_loss = []
        train_dice = []

        for batch in train_loader:  
            with accelerator.accumulate(model):
                steps+=1
                data = batch['img'].to(device,non_blocking=True)
                label = batch['segm'].squeeze().long().to(device,non_blocking=True)

                output = model(data) 
                if args.loss == 'ce_dice':
                    label_one_hot = one_hot(label[:, None, ...], num_classes=2)
                    loss_output = loss_ce(output,label) + loss_dice(output,label_one_hot)
                elif args.loss == 'ce':
                    loss_output = loss_ce(output,label)
                elif args.loss == 'dice':
                    label_one_hot =  one_hot(label[:, None, ...], num_classes=2)
                    loss_output = loss_dice(output,label_one_hot)
                accelerator.backward(loss_output)
                optimizer.step()
                if not args.reduceLROnPlateau:
                    scheduler.step()
                optimizer.zero_grad()
                
            dice_score = diceCoef(output, label)

            train_dice.append(dice_score[0])
            train_loss.append(loss_output.item())

            writer_dict['walltime'].append( time.time() )
            lr = optimizer.param_groups[0]['lr']
            writer_dict['lr'].append(lr)
            writer.add_scalar('utils/lr', lr, steps)
        
        if epoch % args.save_freq == 0:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
                
            gt_image = overlaySegmentation(data, label)
            pred_image = overlaySegmentation(data, torch.argmax(output,dim=1))
            
            grid = make_grid( torch.cat( (gt_image, pred_image), dim=-1 ), nrow=1)        
            writer.add_image('img/train_batch', grid,  steps)


            torch.save({
                'epoch': epoch,
                'save_dir': latest_model_path,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args},
                latest_model_path)
        
        if epoch == args.epochs:
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
    
            torch.save({
                'epoch': epoch,
                'save_dir': final_model_path,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args},
                final_model_path)
        
        model.eval()
     
        valid_loss = []
        valid_dice = []
        
        count=1
        for batch in valid_loader:
            data = batch['img'].to(device,non_blocking=True)
            label = batch['segm'].to(device,non_blocking=True).squeeze().long()
        
            output = model(data)
            if args.loss == 'ce_dice':
                label_one_hot =  one_hot(label[:, None, ...], num_classes=2)
                valid_loss.append((loss_ce(output,label) + loss_dice(output,label_one_hot)).item())
            elif args.loss == 'ce':
                valid_loss.append(loss_ce(output,label).item())
            elif args.loss == 'dice':
                label_one_hot =  one_hot(label[:, None, ...], num_classes=2)
                valid_loss.append(loss_dice(output,label_one_hot).item())
            dice_score = diceCoef(output, label)
            valid_dice.append(dice_score[0])
           
            if epoch % args.save_freq == 0 and count==len(valid_loader)-1:
                gt_image = overlaySegmentation(data, label)
                pred_image = overlaySegmentation(data, torch.argmax(output,dim=1))
                
                grid = make_grid( torch.cat( (gt_image, pred_image), dim=-1 ), nrow=1)        
                writer.add_image('img/valid_batch', grid,  steps)

            count+=1    

           
        if args.reduceLROnPlateau:
            scheduler.step(np.mean(valid_loss))
        
        # EARLY STOPPING 
        if epoch == start_epoch:
            min_valid_loss = np.mean(valid_loss)
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
    
            torch.save({
                'epoch': epoch,
                'save_dir': best_model_path,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args},
                best_model_path)
            
        elif np.mean(valid_loss) < min_valid_loss:
            min_valid_loss = np.mean(valid_loss)
            state_dict = model.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
    
            torch.save({
                'epoch': epoch,
                'save_dir': best_model_path,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args},
                best_model_path)
            
            count_earlyStopping = 0
            
            logging_dict['MinLoss'] = min_valid_loss
            logging_dict['Epoch_bestVal'] = epoch

        else:
            count_earlyStopping += 1
        
        save_obj_pkl(model_path + os.sep +'tensorboard_writer.pkl', writer_dict )       
       
        writer_dict['loss_train'].append(np.mean(train_loss))
        writer_dict['loss_valid'].append(np.mean(valid_loss))
        writer_dict['dice_train'].append(np.mean(train_dice))
        writer_dict['dice_valid'].append(np.mean(valid_dice))
   
        writer.add_scalars('loss', {'train':np.mean(train_loss),
                                    'valid':np.mean(valid_loss)},steps)
        
        writer.add_scalars('dice', {'train': np.mean(train_dice),
                                    'valid': np.mean(valid_dice)},steps)
        
        end_time = time.time()
    
        print('Epoch %03d, time for epoch: %3.2f' %(epoch, end_time-start_time))
        print('loss %2.4f, validation loss %2.4f, train dice %2.4f, val dice %2.4f' %(np.mean(train_loss), np.mean(valid_loss),
                                                                                      np.mean(train_dice), np.mean(valid_dice)))
        
        if args.earlyStopping and count_earlyStopping >= args.earlyStoppingPatience:
         
            print(f'Early stopping after {epoch} epochs')
         
            torch.save({
                'epoch': epoch,
                'save_dir': latest_model_path,
                'state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args},
                latest_model_path)
            
            break
        
    writer.close()  
    
    logging_df = pd.DataFrame(logging_dict,index=[0])
    logging_df.to_csv(f'{model_path}/logging_dict.csv')    

    outputPath_bestModel = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models','eatSegm.model')
    shutil.copy(best_model_path,outputPath_bestModel)

    if args.evalTest:
        df_test = pd.DataFrame([np.array([x['img'],x['segm']]) for x in data_dict_test],columns=['img','segm'])
        output_path_inferTest = os.path.join(args.output_path,'inference','Test')
        if not os.path.exists(output_path_inferTest):
            os.makedirs(output_path_inferTest)
        sliceSegmentator_predSlice = SliceSegmentation(df_test, output_path_inferTest, args.use_cache, args.pin_memory, args.num_workers, 
                                                   args.gpuNum, args.batch_size, args.plotResults)
        inferDict_test = sliceSegmentator_predSlice.segmentSlice()
        
        pd.DataFrame(inferDict_test).to_csv(os.path.join(output_path_inferTest,'results.csv'))

    if args.evalVal:
        df_val = pd.DataFrame([np.array([x['img'],x['segm']]) for x in data_dict_val],columns=['img','segm'])
        output_path_inferVal = os.path.join(args.output_path,'inference','Val')
        if not os.path.exists(output_path_inferVal):
            os.makedirs(output_path_inferVal)
        sliceSegmentator_predSlice = SliceSegmentation(df_val, output_path_inferVal, args.use_cache, args.pin_memory, args.num_workers, 
                                                   args.gpuNum, args.batch_size, args.plotResults)
        inferDict_val = sliceSegmentator_predSlice.segmentSlice()
        pd.DataFrame(inferDict_val).to_csv(os.path.join(output_path_inferVal,'results.csv'))
    
if __name__ == "__main__": 
    main()

