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
import pandas as pd
import re
from utils.calcDistance import calcDistance

def calculate_statistics(data, percentiles = [10, 50, 90]):
    statistics           = data.describe(percentiles=[p / 100 for p in percentiles])
    selected_percentiles = [f'{p}%' for p in percentiles]  
    results              = statistics.loc[['mean', 'std', 'min'] + selected_percentiles + ['max']]
    return results

########################### Outlier Detection Function ###########################
def outlier_detection_ZScore_MAD(data, spacing, threshold = 2.5, mm_border = 15):

    data = data.astype(float)

    if (abs(np.max(data, axis=0) - np.min(data, axis=0))*spacing >mm_border).any(): 

        data_median = np.median(data, axis=0)
        data_med_abs_deviation = abs((data - data_median)*spacing)
        mad  = np.median(data_med_abs_deviation, axis=0) * 0.721347 # 1.4826 #-> normal distribution
        mask = mad != 0
        data_deviation = (data - data_median)*spacing
        
        for i in range(data.shape[0]):
            data_deviation[i,mask] = data_deviation[i,mask]/mad[mask]
            
        mad[mad==0] = 1e-6
        data_deviation = data_deviation/mad
        df_scores_filtered = (data_deviation > -threshold) & (data_deviation < threshold)
        no_outliers = df_scores_filtered.all(axis=1)
        
        if np.all(no_outliers==False):
            no_outliers[:] = True
            
        true_indices = [ind for ind, value in enumerate(no_outliers) if value.any()]   
        num_points = data.shape[0]
        if len(true_indices)<num_points:
            really_no_outliers = []
            for ind in range(num_points):
                if ind not in true_indices and calcDistance(np.mean(data[true_indices]), data[ind], spacing)>mm_border:
                    really_no_outliers.append(False)
                else:
                    really_no_outliers.append(True)
        else:
            really_no_outliers = no_outliers
        
        return data[really_no_outliers], really_no_outliers

    else:

        return data, [num for num in range(data.shape[0])]

        
def do_data_analysis(file_path):

    df         = pd.read_csv(file_path, sep=',', encoding='utf-8')
    df_titel   = df.columns
    last_titel = re.search(r'\d+', df_titel[-1]) 
    
    if last_titel is not None:
        num_agents = int(last_titel.group()) + 1
    else:
        print('No agents number found.')
        sys.exit(1)
            
   ########################### Agents as Dataframes ###########################
    if "111Spacing" in file_path:
        new_titel = ['Filename', 'Agent pos x', 'Agent pos y', 'Agent pos z', 'Landmark pos x', 'Landmark pos y', 'Landmark pos z', 'Spacing', 'Distance']
    else:
        new_titel = ['Filename', 'Agent pos x', 'Agent pos y', 'Agent pos z', 'Landmark pos x', 'Landmark pos y', 'Landmark pos z', 'Spacing', 'Landmark Orig', 'Distance']

    agent_dataframes = {}
    for agents_number in range(num_agents):
        agent_dataframe = df[[column_name for column_name in df.columns if f'{agents_number}' in column_name]]
        agent_dataframe.columns = new_titel
        agent_dataframe.loc[:, 'Filename']  =  agent_dataframe['Filename'].str.split('/').str.get(-1).astype(int)
        agent_dataframes[f'Agent {agents_number}'] = agent_dataframe

    dataframe = {}
    dataframe['Landmark'] = agent_dataframes
            
    ########################### Prepare Dataframe ###########################
        

    data = {}
    for ind in range(len(dataframe['Landmark'])):

        data_agent = dataframe['Landmark']['Agent ' + str(ind)][['Agent pos x', 'Agent pos y', 'Agent pos z']]
        data_agent.columns = ['Agent ' + str(ind) + ' pos x', 'Agent ' + str(ind) + ' pos y', 'Agent ' + str(ind) + ' pos z']
        if ind == 0:
            data = dataframe['Landmark']['Agent ' + str(ind)][['Landmark pos x', 'Landmark pos y', 'Landmark pos z', 'Spacing']]
        data = pd.concat([data, data_agent], axis=1)
    
    ########################### Detect Outliers ########################### 
    spacing = []
    rows, cols = data.shape
    agents = (cols-4)/3
    distances = []
    av_agents = []
    z_distances = []
    datas_without_outliers = []
    no_outliers_inds = []

    for _, row in data.iterrows():
        coordinates = []
        for agent in range(int(agents)):
            coordinates.append(row[4+agent*3:4+agent*3+3])
        spacing = [float(x.replace('(','').replace(')','')) for x in row['Spacing'].split(',')]
        coordinates = np.array(coordinates)
        data_without_outliers, no_outliers = outlier_detection_ZScore_MAD(coordinates, spacing)
        datas_without_outliers.append(data_without_outliers)
        no_outliers_inds.append(no_outliers)
        av_agent = np.array([round(float(value)) for value in np.mean(data_without_outliers, axis=0)])
        landmark = np.array(row[0:3])
        av_agents.append(av_agent)
        
        if not landmark[0] is None:
           distance = calcDistance(av_agent, landmark, spacing)
           z_distance = abs(av_agent[-1]-landmark[-1])*spacing[2]
           distances.append(distance)
           z_distances.append(z_distance)
 
    df['Agents without outliers'] = datas_without_outliers
    df['No outliers indices']     = no_outliers_inds
    df['Averaged Agent without outliers']      = av_agents
   
    if not (data['Landmark pos x'].iloc[0] is None or np.isnan(data['Landmark pos x'].iloc[0])):   
      
        df['Averaged Distance without outliers']   = distances
        df['Averaged z Distance without outliers'] = z_distances

    ########################### Do Statistics ###########################     

        stat   = calculate_statistics(df['Averaged Distance without outliers'])
        print(stat)
        z_stat = calculate_statistics(df['Averaged z Distance without outliers'])
        print(z_stat)
        
        stats_df = pd.DataFrame({
            'Definition': stat.index,
            'Averaged Distance without outliers': stat.values,
            'Averaged z Distance without outliers': z_stat.values
        })
        
        if "111Spacing" in file_path:
            name = os.path.join(os.path.dirname(file_path),"statistics_111Spacing.csv")
        else:
            name = os.path.join(os.path.dirname(file_path),"statistics_origSpacing.csv")
        open(name, 'w').close()
        stats_df.to_csv(name, index=False) 
    

    df.to_csv(file_path)
