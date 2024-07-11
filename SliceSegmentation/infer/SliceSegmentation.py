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
from tqdm import tqdm
import numpy as np
import torch
import nibabel as nib
import os
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ScaleIntensityd)
from monai.networks.nets import UNet
from utils.PlotPredictions import plotSegmPred
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


def postprocess(img, segm, threshLower=-190, threshUpper=-30):
    segm[np.logical_or(img < threshLower, img > threshUpper)] = 0
    
    return segm

def extractValues(img, segm, spacing, threshLower=-190, threshUpper=-30):
     
    segm[np.logical_or(img < threshLower, img > threshUpper)] = 0
    volume = np.min(img) * np.ones(img.shape)
    volume[segm==1] = img[segm==1]
  
    area = np.sum(segm)*spacing[0]*spacing[1]/100
    density = np.mean(volume[segm==1])
    
    return area, density
    
class SliceSegmentation:
    def __init__(self,df,outputPath,use_cache=False,
                 pin_memory=False,num_workers=0,gpuNum=0, batchSize=1, plotResults=False):
        
        self.df = df
        self.outputPath = outputPath
        self.pin_memory = pin_memory
        self.use_cache = use_cache
        self.num_workers = num_workers
        self.gpuNum = gpuNum
        self.batchSize = batchSize
        self.plotResults = plotResults
        
    
    def segmentSlice(self):
                
        if self.gpuNum > torch.cuda.device_count():
            self.gpuNum = torch.cuda.device_count()-1
        
        device = torch.device(f'cuda:{self.gpuNum}' if torch.cuda.is_available() else 'cpu')

        gtAvailable = False
        if 'segm' in self.df.keys(): 
            gtAvailable = True
            data_dicts = np.asarray([
                {'img': img, 'segm': segm}
                for img, segm in 
                zip(self.df['img'].to_numpy(),self.df['segm'].to_numpy())])
            
            eval_transforms = Compose([
                LoadImaged(keys=['img','segm']),
                EnsureTyped(keys=['img','segm'], data_type='tensor'),
                EnsureChannelFirstd(keys=['img','segm']),
                ScaleIntensityd(keys=['img'],minv=0,maxv=1)
            ]) 
            
        else:
            data_dicts = np.asarray([
              {'img': img}
              for img in zip(self.df['img'].to_numpy())])
            
            eval_transforms = Compose([
                LoadImaged(keys=['img']),
                EnsureTyped(keys=['img'], data_type='tensor'),
                EnsureChannelFirstd(keys=['img']),
                ScaleIntensityd(keys=['img'],minv=0,maxv=1)
            ]) 

        if self.use_cache:
            ds = CacheDataset(
                data=data_dicts, transform=eval_transforms,
                cache_rate=1.0, num_workers=self.num_workers,)
        else:
            ds = Dataset(data=data_dicts, transform=eval_transforms)
            
        data_loader = DataLoader(ds, batch_size=self.batchSize,
                                 num_workers=self.num_workers,shuffle=False) 
        
        model = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=2, 
            channels=(64, 128, 256, 512, 1024),
            strides=(2, 2, 2, 2),
            num_res_units=2
        )    
        
        modelPath = os.path.join(parent_directory,'models','eatSegm.model')
        checkpoint = torch.load(modelPath, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
   
        if gtAvailable:
            inference_dict = {'PID': [],
                              'Dice':[],
                              'Org Area (cm2)': [],
                              'Org Mean Density (HU)':[],
                              'Pred Area (cm2)':[],
                              'Pred Mean Density (HU)':[]}
        else:
            inference_dict = {'PID': [],
                              'Pred Area (cm2)': [],
                              'Pred Mean Density (HU)':[]}
      
        filenames=[]

        for batch in tqdm(data_loader):
            data = batch['img'].to(device,non_blocking=True)
            data = torch.transpose(data,3,2)


            if gtAvailable:
                label = batch['segm'].squeeze().cpu().numpy()
 
            filenames+=[os.path.dirname(x).split(os.sep)[-1] for x in batch['img_meta_dict']['filename_or_obj']]
            
            output = model(data)
            output = torch.transpose(output,3,2)
            output = torch.argmax(output,axis=1).cpu().detach().numpy()
          
            for cur_batch in range(output.shape[0]):
                # save segmentations
                nii_path = batch['img_meta_dict']['filename_or_obj'][cur_batch]
                cur_pid = os.path.dirname(nii_path).split(os.sep)[-1]
                outputPath_pred = os.path.join(self.outputPath,cur_pid,f'{cur_pid}_predSegm.nii.gz')
                if not os.path.exists(os.path.dirname(outputPath_pred)):
                    os.makedirs(os.path.dirname(outputPath_pred))
                    
                nii_data = nib.load(nii_path)
                cur_data = nii_data.get_fdata()
                spacing = nii_data.header.get_zooms()
                
                cur_output = postprocess(cur_data, output[cur_batch,:,:].astype(np.short))
                
                nii_pred = nib.Nifti1Image(cur_output,nii_data.affine)
                nii_pred.header.set_sform(nii_data.affine,code=1)
                nii_pred.header.set_qform(nii_data.affine,code=1)
                nib.save(nii_pred,outputPath_pred)
                    
                if self.plotResults:
                    outputPath_plot = os.path.join(self.outputPath,cur_pid,f'{cur_pid}_predSegm.png')
                    fig_results = plotSegmPred(cur_data,cur_output, jetValue = 0.65)
                    fig_results.savefig(outputPath_plot,dpi=300)
                    plt.close(fig_results)
                    
                    if gtAvailable:
                        outputPath_plot = os.path.join(self.outputPath,cur_pid,f'{cur_pid}_gtSegm.png')
                        fig_results = plotSegmPred(cur_data, label[cur_batch,:,:], jetValue = 0.5)
                        fig_results.savefig(outputPath_plot,dpi=300)
                        plt.close(fig_results)
                
                cur_area, cur_density = extractValues(cur_data, cur_output, spacing)
                
                inference_dict['PID'].append(cur_pid)
                inference_dict['Pred Area (cm2)'].append(cur_area)
                inference_dict['Pred Mean Density (HU)'].append(cur_density)

                if gtAvailable:
                    cur_dice = f1_score(label[cur_batch,:,:].squeeze().flatten(),cur_output.flatten())
                    cur_orgArea, cur_orgDensity = extractValues(cur_data, label[cur_batch,:,:].squeeze(),spacing)
                    inference_dict['Dice'].append(cur_dice)
                    inference_dict['Org Area (cm2)'].append(cur_orgArea)
                    inference_dict['Org Mean Density (HU)'].append(cur_orgDensity)
  
        return inference_dict
                
               
