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
import warnings
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils.GetFilesList import getFilesList_fromPath
from utils.ReadData import getData
from utils.transformSpacing import transform_to_targetSpacing

warnings.simplefilter("ignore", category=ResourceWarning)

def preprocess_image(image_file, target_spacing=(1,1,1), landmark_file = None):
    try:
        image, header, landmark_org = getData(image_file, landmark_file)
        orig_spacing = header.get_zooms()
        
        image, spacing, landmark = transform_to_targetSpacing(image, landmark_org, orig_spacing, target_spacing)
        image = image.astype(np.float32)

        image[image<-400] = -400
        image[image>600] = 600
        
        image = (image-np.min(image))/(np.max(image)-np.min(image))
       
        image*=255

        if landmark is not None:
            landmark = np.round(landmark)
        else:
            landmark = np.array([None, None, None])
                
        return image, landmark, spacing, landmark_org
    except Exception as e:
        print(f"Error processing file {image_file}: {e}")
        return None


class LandmarkDataset(Dataset):
    def __init__(self, data_path, num_agents=1, target_spacing=(1,1,1), num_workers=4):
        self.image_files, self.landmark_files = getFilesList_fromPath(data_path)
        self.num_agents = num_agents
        self.target_spacing = target_spacing
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            if len(self.landmark_files)==0:
                self.data = list(tqdm(executor.map(self.preprocess_wrapper, self.image_files)))
            else:
                self.data = list(tqdm(executor.map(self.preprocess_wrapper, self.image_files, self.landmark_files)))
                #self.data = preprocess_image(self.image_files[0], self.target_spacing, self.landmark_files[0])
        self.data = [item for item in self.data if item is not None]
    
    def preprocess_wrapper(self, image_file, landmark_file=None):
        try:
            return preprocess_image(image_file, self.target_spacing, landmark_file)
        except Exception as e:
            print(f"Error processing file {image_file}: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
    
    @property
    def num_files(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        image, landmark, spacing, landmark_org = self.data[idx]
        
        images    = [image] * self.num_agents
        landmarks = [landmark] * self.num_agents if landmark is not None else [None] * self.num_agents
        image_filenames = [self.image_files[idx][:-7]] * self.num_agents 
        
        return images, landmarks, image_filenames, spacing, landmark_org
    
    
class EfficientDataLoader:
    def __init__(self, dataset, shuffle=False):
        self.dataset = dataset
        self.shuffle = shuffle
        self.indices = list(range(self.dataset.num_files))
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_index = 0
   
    def __next__(self):
        if self.current_index >= self.dataset.num_files:
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_index = 0
        
        while True:
            idx = self.indices[self.current_index]
            self.current_index += 1
            item = self.dataset[idx]
            if item is not None:
                return item
            if self.current_index >= len(self.dataset):
                raise StopIteration("No valid data found in the entire dataset")

