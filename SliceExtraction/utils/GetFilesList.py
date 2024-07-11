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

def getFilesList_fromPath(data_path):

    folders = [x[0] for x in os.walk(data_path)][1:]

    img_files_list = []
    landmark_files_list = []

    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                file_path = root + os.sep + os.path.basename(file)#.decode('utf-8')
                if file_path.endswith('.nii.gz') and not file.startswith('.') and not 'segm' in file.lower():
                    img_files_list.append(file_path)
                elif file_path.endswith('.json') and not file.startswith('.') and not 'segm' in file.lower():
                    landmark_files_list.append(file_path)

    return img_files_list, landmark_files_list
