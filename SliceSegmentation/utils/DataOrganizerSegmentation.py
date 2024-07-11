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

import os
import sys
parent_directory = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1])
sys.path.append(parent_directory) 
import numpy as np
import cv2
from scipy.ndimage import label, binary_opening, binary_dilation
from skimage.transform import resize
import copy
import pandas as pd
from tqdm import tqdm 
import nibabel as nib
import slicerio 
import nrrd 
from utils.PlotPredictions import plotSegmPred
from matplotlib import pyplot as plt

def preProcessData(data, segm=None, clip_lower = -400, clip_upper = 600):
    
    if segm is None:
        segm = np.zeros(data.shape)
        
    if data.shape != (512,512):
        data_reshaped = resize(data,(512,512),order=3) # Bi-cubic
        segm_reshaped = resize(segm,(512,512),order=0) # Nearest Neighbour
    else:
        data_reshaped = data 
        segm_reshaped = segm

    data_reshaped[data_reshaped>clip_upper] = clip_upper
    data_reshaped[data_reshaped<clip_lower] = clip_lower

    body_mask = np.zeros(data_reshaped.shape).astype(np.uint8)
    body_mask[data_reshaped > clip_lower] = 1
    
    segmed_arr, num_segms = label(body_mask)

    if num_segms > 2:
       body_mask = np.zeros(data_reshaped.shape).astype(np.uint8)
       sizeElement = []
       for i in range(0,num_segms-1):
           indCursegm = np.where(segmed_arr == i+1)
           sizeElement.append(len(indCursegm[0]))
       biggestComponent = np.argmax(sizeElement)+1
       body_mask[segmed_arr==biggestComponent] = 1
    
    res = cv2.findContours(body_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = res[-2] 

    total_mask_wLung = cv2.fillPoly(body_mask,pts=contours,color=(1,1,1))

    lung_segm = np.zeros(data_reshaped.shape)
    lung_segm[data_reshaped==clip_lower] = 1
    kernel = np.ones((3,3),np.uint8)    
    lung_segm = binary_dilation(lung_segm,kernel,iterations=1) 
    segmed_arr, num_segms = label(lung_segm)
    if num_segms > 2:
        lung_segm = np.zeros(data_reshaped.shape).astype(np.uint8)
        sizeElement = []
        for i in range(0,num_segms):
            indCursegm = np.where(segmed_arr == i+1)
            sizeElement.append(len(indCursegm[0]))
        
        sizeElement_big = [x for x in sizeElement if x > 8000]
        sizeElement_new = copy.deepcopy(sizeElement_big)
        if len(sizeElement_big) > 2:
            while np.max(sizeElement_new) > 80000:
                sizeElement_new.remove(np.max(sizeElement_new))
        
        if len(sizeElement_new) > 2:
            remaining_labels = [sizeElement.index(x)+1 for x in sizeElement_new]
            indices = [np.where(segmed_arr==x)[0] for x in remaining_labels]

            keep_labels = [remaining_labels[x] for x in range(len(remaining_labels)) if 0 not in indices[x] and 511 not in indices[x]]

            if len(keep_labels) != 2:
                sizeElement_new.sort(reverse=True)
                sizeElement_new = sizeElement_new[0:2]
                keep_labels = [sizeElement.index(x)+1 for x in sizeElement_new]
            
            lung_segm[segmed_arr==keep_labels[0]] = 1
            lung_segm[segmed_arr==keep_labels[1]] = 1

        elif len(sizeElement_new) == 2:
            keep_labels = [sizeElement.index(x)+1 for x in sizeElement_new]
            lung_segm[segmed_arr==keep_labels[0]] = 1
            lung_segm[segmed_arr==keep_labels[1]] = 1
        else:
            biggestComponent = sizeElement.index(np.max(sizeElement_new))+1
            lung_segm[segmed_arr==biggestComponent] = 1

    ratio_lungFG = np.sum(total_mask_wLung[lung_segm==1])/(len(np.where(lung_segm==1)[0])+1e6)

    if ratio_lungFG < 0.70:
        img_masked = data_reshaped 
    else:
        kernel = np.ones((3,3),np.uint8)    
        total_mask_wLung = binary_opening(total_mask_wLung,kernel,iterations=1) 
        
        segmed_arr, num_segms = label(total_mask_wLung)
        if num_segms > 2:
            total_mask_wLung = np.zeros(data_reshaped.shape).astype(np.uint8)
            sizeElement = []
            for i in range(0,num_segms-1):
                indCursegm = np.where(segmed_arr == i+1)
                sizeElement.append(len(indCursegm[0]))
            biggestComponent = np.argmax(sizeElement)+1
            total_mask_wLung[segmed_arr==biggestComponent] = 1

        img_masked = clip_lower*np.ones(data_reshaped.shape)
        img_masked[total_mask_wLung==1] = data_reshaped[total_mask_wLung==1]
        
    return img_masked, segm_reshaped 

def splitIntoTrainTestValidation(data,trainPercentage=0.7,valPercentage=0.15):
    numTrain = int(np.round(trainPercentage*len(data)))
    numVal = int(np.round(valPercentage*len(data)))

    data_permute = np.random.permutation(data).tolist()
       
    trainData = data_permute[:numTrain]
    valData = data_permute[numTrain:numTrain+numVal]
    testData = data_permute[numTrain+numVal:]
    trainData.sort()
    valData.sort()
    testData.sort()
        
    return trainData,valData,testData

def splitIntoTrainTestValidationByPredefinedSplitting(data,split_path):     
    table = pd.read_csv(split_path,index_col=0)
    
    trainData = []
    valData = []
    testData = []

    pids = list(np.unique([os.path.dirname(x).split('/')[-1] for x in data]))
    pids.sort()
    
    for cur_pid in pids:
        cur_data_paths = [x for x in data if os.path.dirname(x).split('/')[-1] == cur_pid]

        table_row = list(table.ID).index(int(cur_pid))
        isTest = table.iloc[table_row].isTest
        isVal = table.iloc[table_row].isVal

        if isTest:
            testData += cur_data_paths
        elif isVal:
            valData += cur_data_paths
        else:
            trainData += cur_data_paths
   
    return trainData,valData,testData

def organizeData(data, segm_path, output_path, trainPercentage=None, valPercentage=None, 
                 split_path=None, slicerSegmName=None, visualize=False):
    
    data_dict = {'ID': [],
                 'img': [],
                 'segm': [],
                 'isTest': [],
                 'isVal': []
                 }

    if not os.path.exists(output_path):
        os.makedirs(output_path)
     
    if split_path is None:
        if trainPercentage is None or valPercentage is None:
            raise ValueError('Please set trainPercentage and valPercentage or define a split_path')
        train,val,test = splitIntoTrainTestValidation(data, trainPercentage, valPercentage)
    else:
        train,val,test = splitIntoTrainTestValidationByPredefinedSplitting(data, split_path)
        
    imagesTrPath = os.path.join(output_path,'Data','train')
    imagesValPath = os.path.join(output_path,'Data','val')
    imagesTsPath = os.path.join(output_path,'Data','test')
   
    if not os.path.exists(imagesTrPath):
        os.makedirs(imagesTrPath)
    if not os.path.exists(imagesValPath):
        os.makedirs(imagesValPath)
    if not os.path.exists(imagesTsPath):
        os.makedirs(imagesTsPath)
     
    numExistingPat = 0
    alreadyExistingData = []
    for root_path, dirs, files in os.walk(output_path):
        for name in files:
            if name.endswith('.nii.gz'):
                alreadyExistingData.append(name)
                
    if alreadyExistingData:
        numExistingPat = max([int(x.split('.')[0].split('_')[0]) for x in alreadyExistingData])+1 
        
    orgAndAIPath_dict = {'orgNames': [],
                         'aiNames': []}

    for cur_data in tqdm(data):
        cur_pat_id = os.path.dirname(cur_data).split('/')[-1]
        
        img_nii = nib.load(cur_data)
        img = img_nii.get_fdata()
        
        cur_label_path = [x for x in segm_path if os.path.dirname(x).split('/')[-1] == cur_pat_id]
        if len(cur_label_path)==0:
            raise ValueError(f'Did not found ground truth segmentation for Patient {cur_pat_id}')
        elif len(cur_label_path)>1:
            raise ValueError(f'Found more than one ground truth segmentation for Patient {cur_pat_id}. Please check!')
        else:
            cur_label_path=cur_label_path[0]
            
        if cur_label_path.endswith('.seg.nrrd'):
            segm_info = slicerio.read_segmentation_info(cur_label_path)
            segment_names_to_label_values = [(slicerSegmName,1)]
            voxels,header = nrrd.read(cur_label_path)
            segm, extracted_header = slicerio.extract_segments(voxels, header, segm_info, segment_names_to_label_values)
        elif cur_label_path.endswith('.nii.gz') or cur_label_path.endswith('.nii'):
            segm = nib.load(cur_label_path).get_fdata()
        else:
            raise ValueError('Ground Truth segmentation has to be either a nifti or a .seg.nrrd file.')
            
        av_slice = list(np.unique(np.where(segm==1)[2]))
        if len(av_slice) != 1:
            raise ValueError(f'No unique AV slice for {cur_pat_id}')
        av_slice = av_slice[0]
        
        segm_av = segm[:,:,av_slice].squeeze()
        img_av = img[:,:,av_slice].squeeze()

        spacing = img_nii.header.get_zooms()
        
        if int(cur_pat_id) > numExistingPat:
            cur_patName = str(cur_pat_id)
        else:
            cur_patName = f'{int(cur_pat_id)+numExistingPat}'
            numExistingPat+=1
            
        # Check if test, train or validation case
        testCase = 0
        valCase = 0
        if [x for x in test if os.path.dirname(x).split('/')[-1] == cur_pat_id]: testCase = 1
        elif [x for x in val if os.path.dirname(x).split('/')[-1] == cur_pat_id]: valCase = 1
        
        if testCase: 
            cur_save_path = os.path.join(imagesTsPath,cur_patName) 
        elif valCase:
            cur_save_path = os.path.join(imagesValPath,cur_patName) 
        else:
            cur_save_path = os.path.join(imagesTrPath,cur_patName) 

        if not os.path.exists(cur_save_path):
            os.makedirs(cur_save_path)
        
        orgAndAIPath_dict['orgNames'].append(cur_data)
        orgAndAIPath_dict['aiNames'].append(cur_save_path)
        
        cur_output_path_img = os.path.join(cur_save_path,f'{cur_patName}.nii.gz')
        cur_output_path_segm = os.path.join(cur_save_path,f'{cur_patName}_segm.nii.gz')
        
        data_dict['ID'].append(cur_pat_id)
        data_dict['img'].append(cur_output_path_img)
        data_dict['segm'].append(cur_output_path_segm)
        data_dict['isTest'].append(testCase)
        data_dict['isVal'].append(valCase)

        img_av_preProcessed, segm_av_preProcessed = preProcessData(img_av, segm_av)
        
        if visualize:
            av_plot = plotSegmPred(img_av_preProcessed, segm_av_preProcessed, jetValue=0.5)
            av_plot.savefig(cur_output_path_img.replace('.nii.gz','.png'),dpi=300)
            plt.close(av_plot)
    
        # ---------------------------- Write Nifti Image --------------------------
        factor = img_av.shape[0]/512
        new_spacing = [spacing[0]*factor,spacing[1]*factor,1]

        new_affine_av = nib.affines.rescale_affine(img_nii.affine,(*img_av.shape,1),new_spacing,(*img_av_preProcessed.shape,1))
        av_nii_new = nib.Nifti1Image(img_av_preProcessed,new_affine_av)
        av_nii_new.header.set_sform(new_affine_av,code=1)
        av_nii_new.header.set_qform(new_affine_av,code=1)
        nib.save(av_nii_new,cur_output_path_img)
        
        segm_av_nii_new = nib.Nifti1Image(segm_av_preProcessed,new_affine_av)
        segm_av_nii_new.header.set_sform(new_affine_av,code=1)
        segm_av_nii_new.header.set_qform(new_affine_av,code=1)
        nib.save(segm_av_nii_new,cur_output_path_segm)
    
    df = pd.DataFrame.from_dict(data_dict)

    df_orgAndAIPath = pd.DataFrame.from_dict(orgAndAIPath_dict)

    return df, df_orgAndAIPath
     
       
