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

import torch

def diceCoef(pred, label):
    
    num_classes = len(torch.unique(label))
    dice_score_per_class = torch.zeros(num_classes-1)
    
    pred_segm = torch.argmax(pred,dim=1)
    
    for c in range(1,num_classes): # start at 1 as 0 is background
        cur_label = torch.zeros(label.shape)
        cur_label[label==c] = 1
        cur_pred = torch.zeros(pred_segm.shape)
        cur_pred[pred_segm==c] = 1
    
        sum_pred_gt = cur_label + cur_pred
        diff_pred_gt = cur_pred - cur_label 
        
        tp = len(torch.where(sum_pred_gt==2)[0])
        fn = len(torch.where(diff_pred_gt==-1)[0])
        fp = len(torch.where(diff_pred_gt==1)[0])
       
        cur_dice = 2*tp/(2*tp+fn+fp+1e-10)

        dice_score_per_class[c-1] = cur_dice
        
    return dice_score_per_class
