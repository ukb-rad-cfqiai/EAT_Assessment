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
from infer.evaluator import Evaluator
from utils.logger import Logger
from utils.DQNModel import DQN
from utils.medical import MedicalPlayer, FrameStack
from utils.analysis import do_data_analysis
import argparse
import torch
import numpy as np
import warnings


def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE    = (45, 45, 45)
FRAME_HISTORY = 4

def get_player(directory=None, files_list=None, landmark_ids=None, viz=False,
               task="play", saveGif=False, saveVideo=False,
               multiscale=True, history_length=20, agents=1, logger=None, norm_spacing=True):
    env = MedicalPlayer(
        directory      = directory,
        screen_dims    = IMAGE_SIZE,
        viz            = viz,
        saveGif        = saveGif,
        saveVideo      = saveVideo,
        task           = task,
        files_list     = files_list,
        landmark_ids   = landmark_ids,
        history_length = history_length,
        multiscale     = multiscale,
        agents         = agents,
        logger         = logger,
        norm_spacing   = norm_spacing )
    if task != "train":
        env = FrameStack(env, FRAME_HISTORY, agents)
    return env

def set_reproducible(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--load', help='Path to the model to load',required=True, type=str)
    parser.add_argument('--files', help='Filepath to the directory containing directories of images+landmarks', required=True, type=str)
    parser.add_argument('--normalised_spacing_wanted_for_proceeding', help='Transformas input data into 1x1x1mm^3 spacing, if True', default=True)
    parser.add_argument('--saveGif', help='Save gif image of the game', action='store_true', default=False)
    parser.add_argument('--saveVideo', help='Save video of the game', action='store_true', default=False)
    parser.add_argument('--log_dir', help='Store results in this directory.', default=os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1]), type=str)
    parser.add_argument('--output_dir_name', help='Directory name for storing results.', required=True, type=str)
    parser.add_argument('--log_comment', help='Suffix (current time) appended to the name of the log folder name.',default='', type=str)
    parser.add_argument('--agents', help='Number of agents detecting the landmark', type=int, default=5)
    parser.add_argument('--discount', help='Discount factor used in the Bellman equation', default=0.9, type=float)
    parser.add_argument('--steps_per_episode', help='Maximum steps per episode', default=300, type=int)
    parser.add_argument('--save_freq', help='Saves network every save_freq steps', default=10, type=int)
    parser.add_argument('--viz', help='Size of the window, None for no visualisation', default=0.01, type=float) 
    parser.add_argument('--multiscale', help='Reduces size of voxel around the agent when it oscillates', dest='multiscale', action='store_true', default='False')
    parser.add_argument('--write', help='Saves the training logs', dest='write', action='store_true', default=True)
    parser.add_argument('--team_reward', help='Refers to adding the (potentially weighted) average reward of all agents to their individiual rewards', choices=[None, 'mean', 'attention'], default='mean')
    parser.add_argument('--attention', help='Use attention for communication channel in C-MARL/CommNet', dest='attention', action='store_true', default=False)
    parser.add_argument('--seed', help="Random seed for both training and evaluating. If none is provided, no seed will be set", type=int)
    parser.add_argument('--fixed_spawn', nargs='*', type=float, help='Starting position of the agents during rollout. Randomised if not specified.',)
    parser.add_argument('--task', help='''Task to perform, must load a pretrained model"''', choices=['play', 'eval'], default='eval')
    args = parser.parse_args()
    
    task = args.task
    assert args.agents > 0


    if args.seed is not None:
        set_reproducible(args.seed)

    logger       = Logger(args.log_dir, args.output_dir_name, args.write, args.save_freq, comment=args.log_comment, normalized_scale = args.normalised_spacing_wanted_for_proceeding)
    landmark_ids = [0 for _ in range(args.agents)]

    dqn   = DQN(args.agents, frame_history=FRAME_HISTORY, logger=logger, collective_rewards=args.team_reward, attention=args.attention)
    model = dqn.q_network
    model.load_state_dict(torch.load(args.load, map_location=model.device))
    print('Data preparation in progress ...')
    environment = get_player(files_list   = args.files,
                             landmark_ids = landmark_ids,
                             saveGif      = args.saveGif,
                             saveVideo    = args.saveVideo,
                             task         = task,
                             agents       = args.agents,
                             viz          = args.viz,
                             logger       = logger,
                             norm_spacing = args.normalised_spacing_wanted_for_proceeding)
    evaluator = Evaluator(environment, model, logger, args.agents, args.steps_per_episode)
    evaluator.play_n_episodes(fixed_spawn = args.fixed_spawn)
    
    
    if args.write == True:
        inference_results = os.path.join(logger.dir, 'results_origSpacing.csv')
        print('Statistical analysis of the results in original spacing:')
        do_data_analysis(inference_results)
        if args.normalised_spacing_wanted_for_proceeding:
            print('Statistical analysis of the results in 1*1*1mm³ spacing:')
            inference_results = os.path.join(logger.dir, 'results_111Spacing.csv')
            do_data_analysis(inference_results)
