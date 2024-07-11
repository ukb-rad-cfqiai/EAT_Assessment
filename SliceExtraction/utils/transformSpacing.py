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
import nibabel as nib
from skimage.transform import resize


def getScaleFactor(orig_spacing, target_spacing):
    orig_spacing   = np.asarray(orig_spacing)
    target_spacing = np.asarray(target_spacing)
    return orig_spacing/target_spacing

def doTransforms(img, landmark, scaleFactor):
    if img is not None:
        transformed_img = resize(img, img.shape*scaleFactor, order=3, anti_aliasing=False, mode='constant', cval=0)
    else:
        transformed_img = None
        
    if landmark is not None and not None in landmark:
        transformed_landmark = landmark * scaleFactor
    else: 
        transformed_landmark = landmark
        
    return transformed_img, transformed_landmark

def transform_to_targetSpacing(img, landmark, orig_spacing, target_spacing, img_path = None, points_only = False):
              
    if orig_spacing is None or target_spacing is None:
        niiFile = nib.load(img_path)
        
        if orig_spacing is None and target_spacing is None:
            raise(ValueError('Neither orig_spacing nor target_spacing are given to do transforms!'))
        elif orig_spacing is None:
            orig_spacing = niiFile.header.get_zooms()
        elif target_spacing is None:
            target_spacing = niiFile.header.get_zooms() 
        
        if orig_spacing != target_spacing:
            if not points_only:
                img     = niiFile.get_fdata()
            scaleFactor   = getScaleFactor(orig_spacing, target_spacing)
            img, landmark = doTransforms(img, landmark, scaleFactor)
            
    else: 
        if any(np.asarray(orig_spacing) != np.asarray(target_spacing)):
            scaleFactor   = getScaleFactor(orig_spacing, target_spacing)
            img, landmark = doTransforms(img, landmark, scaleFactor)
        else:
            target_spacing = orig_spacing
            
    return img, target_spacing, landmark
