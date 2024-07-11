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

import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy

def plotSlicePred(image,coord_gt=None,coord_pred=None,coord_color_gt='lime',coord_color_pred='cyan',plane='sagittal',resolutionsRatioZ=1,f=None):
    
    image_plot = copy.deepcopy(image)
    
    if not f:
        f = plt.figure()
    
    window_w = 600
    window_l = 125
    hu_lower = int(np.round(window_l - (window_w/2)))
    hu_upper = int(np.round(window_l + (window_w/2)))

    image_plot[image_plot < hu_lower] = hu_lower
    image_plot[image_plot > hu_upper] = hu_upper

    exist_coord = True
    if coord_gt is None and coord_pred is None:
        exist_coord = False
        coord = np.round(np.array(image_plot.shape)/2).astype(int)
    elif coord_gt is None:
        coord = coord_pred
    else:
        coord = coord_gt

    image_plot = np.squeeze(image_plot)
    
    if len(np.shape(image_plot)) == 4: #take first channel
        image_plot = image_plot[0,:]
     
    if plane == 'axial':
        image_plot = image_plot[:,:,coord[2]]
    elif plane == 'coronal':
        image_plot = image_plot[:,coord[1],:]
    elif plane == 'sagittal':
        image_plot = image_plot[coord[0],:,:]
    
    image_plot = np.flip(image_plot,1)  
    image_plot = image_plot.transpose((1, 0))  

    if not resolutionsRatioZ == 1:
        if plane == 'sagittal' or plane == 'coronal':
            image_shape = np.asarray(image_plot.shape)
            image_shape[0] *= resolutionsRatioZ
            if exist_coord:
                coord = [coord[0],coord[1],int(np.round(coord[2]*resolutionsRatioZ))]
                if not coord_gt is None:
                    coord_gt = [coord_gt[0],coord_gt[1],int(np.round(coord_gt[2]*resolutionsRatioZ))]
                if not coord_pred is None:
                    coord_pred = [coord_pred[0],coord_pred[1],int(np.round(coord_pred[2]*resolutionsRatioZ))]

            width = image_shape[1]
            height = image_shape[0]
            dim = [width,height]
            image_plot = cv2.resize(image_plot,tuple(dim))
    
    plt.imshow(image_plot,cmap='gray')
    if exist_coord:
        if plane == 'axial':
            if not coord_gt is None:
                plt.scatter(coord_gt[0],image_plot.shape[1]-coord_gt[1],color=coord_color_gt,marker='o',s=20)
            if not coord_pred is None:
                plt.scatter(coord_pred[0],image_plot.shape[1]-coord_pred[1],color=coord_color_pred,marker='o',s=20)
        elif plane == 'sagittal':
            if not coord_gt is None:
                plt.scatter(coord_gt[1],image_plot.shape[0]-coord_gt[2],color=coord_color_gt,marker='o',s=20)
                plt.axhline(image_plot.shape[0]-coord_gt[2],color=coord_color_gt)
            if not coord_pred is None:
                plt.scatter(coord_pred[1],image_plot.shape[0]-coord_pred[2],color=coord_color_pred,marker='o',s=20)
                plt.axhline(image_plot.shape[0]-coord_pred[2],color=coord_color_pred)

        else: # coronal
            if not coord_gt is None:                
                plt.scatter(coord_gt[0],image_plot.shape[0]-coord_gt[2],color=coord_color_gt,marker='o',s=20)
                plt.axhline(image_plot.shape[0]-coord_gt[2],color=coord_color_gt)

            if not coord_pred is None:
                plt.scatter(coord_pred[0],image_plot.shape[0]-coord_pred[2],color=coord_color_pred,marker='o',s=20)
                plt.axhline(image_plot.shape[0]-coord_pred[2],color=coord_color_pred)

            
    plt.axis('off')

    return f

def plotSegmPred(image,label=None,jetValue=0.75,f=None):
    
    image_plot = copy.deepcopy(image)
    
    window_w = 600
    window_l = 125
    hu_lower = int(np.round(window_l - (window_w/2)))
    hu_upper = int(np.round(window_l + (window_w/2)))

    image_plot[image_plot < hu_lower] = hu_lower
    image_plot[image_plot > hu_upper] = hu_upper

    if not f:
        f = plt.figure()

    image_plot = np.squeeze(image_plot)
    
    if label is None:
        label = np.zeros(image_plot.shape)
    else:
        label_plot = copy.deepcopy(label)
    
    if len(np.shape(image_plot)) == 4: #take first channel
        image_plot = image_plot[0,:]
        
    image_plot = np.flip(image_plot,1)  
    label_plot = np.flip(label_plot,1)  
    
    image_plot = image_plot.transpose((1,0))
    label_plot = label_plot.transpose((1,0))
    
    imageForHeatmap = label_plot.astype(float)
    imageForHeatmap[label_plot == 1.0] = jetValue
    
    image_plot = cv2.normalize(image_plot, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    image_plot = image_plot.astype(np.uint8)        
    
    im = np.zeros( np.shape(image_plot) +(3,)).astype(np.uint8)
    im[:,:,0] = image_plot
    im[:,:,1] = im[:,:,0]
    im[:,:,2] = im[:,:,0]
    
    maskLabel = np.zeros( np.shape(im)).astype(bool)
    maskLabel[:,:,0] = imageForHeatmap > 0
    maskLabel[:,:,1] = maskLabel[:,:,0]
    maskLabel[:,:,2] = maskLabel[:,:,0]

    imageForHeatmap = imageForHeatmap * 255.0
    imageForHeatmap = imageForHeatmap.astype(np.uint8)
    heatmapImage = cv2.applyColorMap(imageForHeatmap, cv2.COLORMAP_JET).astype(float)
    
    maskBg = np.zeros(np.shape(im)).astype(bool)
    maskBg[maskLabel==False]=True
    heatmapImage = cv2.addWeighted(heatmapImage, 0.4, im , 0.6, 0, dtype = cv2.CV_8UC1)
    heatmapImage[maskBg] = im[maskBg]
    
    plt.imshow(heatmapImage).set_cmap('jet')

    plt.axis('off')

    return f
