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
import pandas as pd
from utils.DataOrganizerSegmentation import preProcessData
from infer.SliceSegmentation import SliceSegmentation, extractValues
from utils.PlotPredictions import plotSlicePred, plotSegmPred
from utils.ReadGTCoord import get_label_fromJson
import slicerio
import nrrd
import nibabel as nib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="path to input data", type=str, required=True)
    parser.add_argument("--sliceExtractionRes", help='Path to slice extraction', type=str, required=True)
    parser.add_argument("--output_path", help="path where to save output data", type=str, required=True)
    parser.add_argument("--gtAvailable", help='Use this if ground truth segmentations are available and you want to evaluate performance',required=False, action="store_true")
    parser.add_argument("--gtSegmIdentifier", help="Specify file ending of ground truth segmentations (e.g. Segm.nii.gz, Segm.nii or Segm.seg.nrrd). Default: segm.nii.gz", type=str, required=False, default='segm.nii.gz')
    parser.add_argument("--slicerSegmName", help="Specify segmetation name for segmentations from 3D Slicer. Only relevant if gtSegmIdentifier == .seg.nrrd", type=str, required=False)
    parser.add_argument("--plotResults", help="Use this if you want to visualize output predictions", required=False, action="store_true")
    parser.add_argument("--batchSize", help="Set batch size for inference, default: 10", required=False, type=int, default=10)
    parser.add_argument("--num_workers", help="Set number of workers used for data loading, default: 4", type=int, default=4)
    parser.add_argument("--pin_memory", help='Use this if you want to pin memory on gpu',required=False, action="store_true")
    parser.add_argument("--use_cache", help='Use this if you want to use cache',required=False, action="store_true")
    parser.add_argument("--gpuNum", help='Set number of gpu you want to use, default:0', required=False, type=int, default=0)

    args = parser.parse_args()
    
    args.gtAvailable = True
    args.plotResults = True
    
    result_dict = {'ID':[],
                   'img':[],
                   'AV_x_pred':[],
                   'AV_y_pred':[],
                   'AV_z_pred':[]
                   }
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    data = []
    for root_path, dirs, files in os.walk(args.data_path):
        for name in files:
            if (name.endswith('.nii.gz') and not name.startswith('.')) and not args.gtSegmIdentifier in name:
                data.append(os.path.join(root_path,name))
    data.sort()
    
    print(f'-------------- Found {len(data)} input files ---------------------')

    res_sliceExtraction = pd.read_csv(args.sliceExtractionRes, index_col=0)
        
    if args.gtAvailable:
        result_dict['AV_x_GT'] = []
        result_dict['AV_y_GT'] = []
        result_dict['AV_z_GT'] = []
        result_dict['GT Area (cm2)'] = []
        result_dict['GT Mean Density (HU)'] = []

    
    print ('------------------- Preprocess for EAT segmentation ----------------------')
    for cur_data in tqdm(data):
        
        cur_id = os.path.basename(cur_data).split('.')[0]

        cur_output_path = os.path.join(args.output_path,cur_id)
        if not os.path.exists(cur_output_path):
            os.makedirs(cur_output_path)

        result_dict['ID'].append(cur_id)
        tabEntry_sliceExtraction = res_sliceExtraction.iloc[[x for x in range(len(res_sliceExtraction)) if os.path.basename(res_sliceExtraction['Filepath 0'].iloc[x]) == cur_id]]
        
        pred_coord = tabEntry_sliceExtraction['Averaged Agent without outliers'].iloc[0]
        pred_coord = pred_coord.replace('[','').replace(']','').split(' ')
        pred_coord = [int(x) for x in pred_coord if x.isdigit()]
        
        nii_data_org_path = os.path.join(args.data_path,cur_id,f'{cur_id}.nii.gz')
        nii_data_org = nib.load(nii_data_org_path)
        data_org = nii_data_org.get_fdata()
        
        org_spacing = nii_data_org.header.get_zooms()
         
        if args.gtAvailable:
            org_coord_path = os.listdir(os.path.join(args.data_path,cur_id))
            org_coord_path = [x for x in org_coord_path if x.endswith('.json') and not x.startswith('.')]
    
            if len(org_coord_path) != 1:
                raise ValueError(f'Could not find unique ground truth coordinate for patient {cur_id}')
        
            org_coord_path = os.path.join(args.data_path,cur_id,org_coord_path[0])
            coord_gt = get_label_fromJson(org_coord_path, nii_data_org)
            
            org_segm_path = os.listdir(os.path.join(args.data_path,cur_id))
            org_segm_path = [x for x in org_segm_path if x.endswith(args.gtSegmIdentifier) and not x.startswith('.')]
            
            if len(org_segm_path) != 1:
                raise ValueError(f'Could not find unique ground truth segmentation for patient {cur_id}')
        
            org_segm_path = os.path.join(args.data_path,cur_id,org_segm_path[0])
            
            if args.gtSegmIdentifier.endswith('.seg.nrrd'):
                segm_info = slicerio.read_segmentation_info(org_segm_path)
                segment_names_to_label_values = [(args.slicerSegmName,1)]
                voxels,header = nrrd.read(org_segm_path)
                segm, extracted_header = slicerio.extract_segments(voxels, header, segm_info, segment_names_to_label_values)
            elif args.gtSegmIdentifier.endswith('.nii.gz') or args.gtSegmIdentifier.endswith('.nii'):
                segm = nib.load(org_segm_path).get_fdata()
            else:
                raise ValueError('Ground Truth segmentation has to be either a nifti or a .seg.nrrd file.')
                
            av_slice = list(np.unique(np.where(segm==1)[2]))
            if len(av_slice) != 1:
                raise ValueError(f'No unique AV slice for {cur_id}')
            av_slice = av_slice[0]
            if av_slice != coord_gt[2]:
                print (f'Warning! z-Coordinate for AV Slice does not match for {cur_id}')
                coord_gt[2] = av_slice
            
            result_dict['AV_x_GT'].append(coord_gt[0])
            result_dict['AV_y_GT'].append(coord_gt[1])
            result_dict['AV_z_GT'].append(coord_gt[2])
            
            segm_av = segm[:,:,av_slice].squeeze()
            data_org_av = data_org[:,:,av_slice].squeeze()
            
            data_org_av_preprocessed, segm_av_preprocessed = preProcessData(data_org_av,segm_av)
            factor = data_org_av.shape[0]/512
            new_spacing = [org_spacing[0]*factor,org_spacing[1]*factor,1]
        
            new_affine_av = nib.affines.rescale_affine(nii_data_org.affine,(*data_org_av.shape,1),new_spacing,(*data_org_av_preprocessed.shape,1))
            av_nii_new = nib.Nifti1Image(data_org_av_preprocessed,new_affine_av)
            av_nii_new.header.set_sform(new_affine_av,code=1)
            av_nii_new.header.set_qform(new_affine_av,code=1)
            nib.save(av_nii_new,os.path.join(cur_output_path,f'{cur_id}_gtAV.nii.gz'))
            
            segm_nii_new = nib.Nifti1Image(segm_av,new_affine_av)
            segm_nii_new.header.set_sform(new_affine_av,code=1)
            segm_nii_new.header.set_qform(new_affine_av,code=1)
            nib.save(segm_nii_new,os.path.join(cur_output_path,f'{cur_id}_gtSegm.nii.gz'))
            
            if args.plotResults:
                output_path_segmGT = os.path.join(cur_output_path,f'{cur_id}_gtSegm.png')
                fig_segmGT = plotSegmPred(data_org_av_preprocessed, segm_av_preprocessed, jetValue = 0.5)
                fig_segmGT.savefig(output_path_segmGT,dpi=300)
                plt.close(fig_segmGT)
            
            gtArea, gtDensity = extractValues(data_org_av_preprocessed, segm_av_preprocessed, new_spacing)
            result_dict['GT Area (cm2)'].append(gtArea)
            result_dict['GT Mean Density (HU)'].append(gtDensity)

        
       
        result_dict['AV_x_pred'].append(pred_coord[0])
        result_dict['AV_y_pred'].append(pred_coord[1])
        result_dict['AV_z_pred'].append(pred_coord[2])

        data_pred_av = data_org[:,:,pred_coord[2]].squeeze()
   
        if args.plotResults:
            output_path_sliceExtraction_pred = os.path.join(cur_output_path,f'{cur_id}_SliceExtraction.png')
            
            if not args.gtAvailable:
                resolutionRatioZ = org_spacing[2]/org_spacing[1]
                fig_sag = plotSlicePred(data_org,None,pred_coord,resolutionsRatioZ=resolutionRatioZ)
                fig_sag.savefig(output_path_sliceExtraction_pred,dpi=300)
                plt.close(fig_sag)

            else:
                resolutionRatioZ = org_spacing[2]/org_spacing[1]
                fig_sag = plotSlicePred(data_org,coord_gt,pred_coord,resolutionsRatioZ=resolutionRatioZ)
                fig_sag.savefig(output_path_sliceExtraction_pred,dpi=300)
                plt.close(fig_sag)
         
        data_pred_av_preprocessed, _ = preProcessData(data_pred_av)
        factor = data_pred_av.shape[0]/512
        new_spacing = [org_spacing[0]*factor,org_spacing[1]*factor,1]
    
        new_affine_av = nib.affines.rescale_affine(nii_data_org.affine,(*data_pred_av.shape,1),new_spacing,(*data_pred_av_preprocessed.shape,1))
        av_nii_new = nib.Nifti1Image(data_pred_av_preprocessed,new_affine_av)
        av_nii_new.header.set_sform(new_affine_av,code=1)
        av_nii_new.header.set_qform(new_affine_av,code=1)
        nib.save(av_nii_new,os.path.join(cur_output_path,f'{cur_id}_predAV.nii.gz'))
        
        result_dict['img'].append(os.path.join(cur_output_path,f'{cur_id}_predAV.nii.gz'))
     
        df_result = pd.DataFrame.from_dict(result_dict)
        df_result.to_csv(os.path.join(args.output_path,'results.csv'))
              
    print('-------------------------- Start EAT Segmentation  --------------------------')
    sliceSegmentator_predSlice = SliceSegmentation(df_result, args.output_path, args.use_cache, args.pin_memory, args.num_workers, 
                                                   args.gpuNum, args.batchSize, args.plotResults)
    inferDict_predAV = sliceSegmentator_predSlice.segmentSlice()
    
    # Add inference results to result dictionary
    result_dict['Pred Area (cm2)'] = []
    result_dict['Pred Mean Density (HU)'] = []
    
    for cur_id in result_dict['ID']:
        cur_ind = inferDict_predAV['PID'].index(str(cur_id))
        result_dict['Pred Area (cm2)'].append(inferDict_predAV['Pred Area (cm2)'][cur_ind])
        result_dict['Pred Mean Density (HU)'].append(inferDict_predAV['Pred Mean Density (HU)'][cur_ind])
    
    df_result = pd.DataFrame.from_dict(result_dict)
    df_result.to_csv(os.path.join(args.output_path,'results.csv')) 
    
if __name__ == "__main__":
    main()

