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
from utils.ReadGTCoord import get_label_fromJson 
import nibabel as nib
import numpy as np


def getData(image_file, keypoint_file):
    try:
        nii_image  = nib.load(image_file)
        header     = nii_image.header
        image_data = nii_image.get_fdata().astype(np.float32)
        
        if keypoint_file is not None:
            keypoint = np.array(get_label_fromJson(keypoint_file, nii_image))
        else:
            keypoint = None
        
        return image_data, header, keypoint
    except Exception as e:
        print(f"Error reading file {image_file}: {e}")
        return None, None, None
