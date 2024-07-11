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

import json
import numpy as np

def get_label_fromJson(labelJson, img_nii):
        
    with open(labelJson, 'r') as f:
        cur_label = json.load(f)
         
    orientation_av = np.asarray([cur_label['markups'][0]['controlPoints'][0]['orientation'][0],cur_label['markups'][0]['controlPoints'][0]['orientation'][4],cur_label['markups'][0]['controlPoints'][0]['orientation'][-1]])
    coord_av_lps = np.asarray(cur_label['markups'][0]['controlPoints'][0]['position']) * orientation_av
    coord_av_ijk  = np.round(np.linalg.inv(img_nii.get_qform()).dot(np.asarray((coord_av_lps.tolist() + [1]))))
   
    f_coord_av_ijk = np.asarray([coord_av_ijk[0],coord_av_ijk[1],coord_av_ijk[2]]).astype(int)
    
    return f_coord_av_ijk
