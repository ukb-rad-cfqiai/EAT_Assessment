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
import torch
from torch.utils.tensorboard import SummaryWriter
import csv
from utils.transformSpacing import transform_to_targetSpacing
from utils.calcDistance import calcDistance


class Logger(object):
    def __init__(self, output_path, directory_name, write, save_freq=10, comment="", normalized_scale = True):
        self.directory_name = directory_name
        self.output_path    = output_path
        self.write          = write
        self.fig_index      = 0
        self.model_index    = 0
        self.norm_scale     = normalized_scale
        self.save_freq      = save_freq
        if self.write:
            self.output_dir  = os.path.join(self.output_path, self.directory_name)
            os.makedirs(self.output_dir, exist_ok=True)
            self.boardWriter = SummaryWriter(log_dir=self.output_dir, comment=comment)
            self.dir         = self.output_dir


    def write_to_board(self, name, scalars, index=0):
        self.log(f"{name} at {index}: {str(scalars)}")
        if self.write:
            for key, value in scalars.items():
                self.boardWriter.add_scalar(f"{name}/{key}", value, index)

    def plot_res(self, losses, distances):
        if len(losses) == 0 or not self.write:
            return
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2)
        axs[0].plot(list(range(len(losses))), losses, color='orange')
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("Loss")
        axs[0].set_title("Training")
        axs[0].set_yscale('log')
        for dist in distances:
            axs[1].plot(list(range(len(dist))), dist)
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("Distance change")
        axs[1].set_title("Training")

        if self.fig_index > 0:
            os.remove(os.path.join(self.dir, f"res{self.fig_index-1}.png"))
        fig.savefig(os.path.join(self.dir, f"res{self.fig_index}.png"))
        self.boardWriter.add_figure(f"res{self.fig_index}", fig)
        self.fig_index += 1

    def log(self, message, step=0):
        print(str(message))
        if self.write:
            with open(os.path.join(self.dir, "logs.txt"), "a") as logs:
                logs.write(str(message) + "\n")

    def save_model(self, state_dict, name="dqn.pt", forced=False):
        if not self.write:
            return
        if (forced or
           (self.model_index > 0 and self.model_index % self.save_freq == 0)):
            torch.save(state_dict, os.path.join(self.dir, name))

    def write_locations(self, row):
        self.log(str(row))
        if isinstance(row[0], str):
            is_header = True
            self.num_agents = int(row[-1].split(' ')[-1])+1
        else:
            is_header = False
        org_landmark = row[9] 
        
        if not isinstance(org_landmark, list) and org_landmark is None:
            org_landmark = [None, None, None]
        indices_org_landmark = [(i+9)+i*9 for i in range(self.num_agents)]
        
        row_temp = [row[x] for x in range(len(row)) if x not in indices_org_landmark]
        
        if self.write:
            if self.norm_scale: 
                with open(os.path.join(self.dir, 'results_111Spacing.csv'), mode='a', newline='') as f:
                    res_writer = csv.writer(f)
                    res_writer.writerow(row_temp)
            with open(os.path.join(self.dir, 'results_origSpacing.csv'), mode='a', newline='') as ff:
                res_writer_orig = csv.writer(ff)
                row_orig = row
                if not is_header:
                    row_orig = row
                    for i in range(self.num_agents):
                        ind = i*10 + 2
                        _, new_spacing, transformed_agents = transform_to_targetSpacing(None, row_orig[ind:ind+3], orig_spacing = (1,1,1), target_spacing = None, img_path = (row_orig[ind-1] + '.nii.gz'), points_only = True)
                        row_orig[ind+6] = new_spacing
                        row_orig[ind :ind+3] = [int(round(x)) for x in transformed_agents] 
                        #_, _, transformed_landmarks = transform_to_targetSpacing(None, row_orig[ind+4:ind+7], orig_spacing = (1,1,1), target_spacing = None, img_path = (row_orig[ind] + '.nii.gz'), points_only = True)
                        row_orig[ind+3 :ind+6] = [int(x) if x is not None else None for x in org_landmark]#[int(round(x)) if x is not None else None for x in transformed_landmarks]
                        if org_landmark[0] is None:
                            row_orig[ind+8] = None
                        else:
                            row_orig[ind+8] = calcDistance(transformed_agents, org_landmark, new_spacing)
                res_writer_orig.writerow(row_orig)
