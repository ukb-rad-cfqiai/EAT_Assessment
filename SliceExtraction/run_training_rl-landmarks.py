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
import warnings
from utils.logger import Logger
from training.trainer import Trainer
from utils.medical import MedicalPlayer, FrameStack
import argparse
import torch
import numpy as np

def warn(*args, **kwargs):
    pass

warnings.warn = warn
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

IMAGE_SIZE = (45, 45, 45)
# observations the network can see:
FRAME_HISTORY = 4

def get_player(directory=None, files_list=None, landmark_ids=None, viz=False,
               task="play", saveGif=False, saveVideo=False, norm_spacing = True,
               multiscale=True, history_length=20, agents=1, logger=None):
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
        norm_spacing   = norm_spacing)
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

    parser.add_argument('--files', help='Filepath to the directory containing directories of images and landmarks', required=True, type=str)
    parser.add_argument('--val_files', help='Filepath to the directory containing directories of images and landmarks for evaluation', required=False)
    parser.add_argument('--normalised_spacing_wanted_for_proceeding', help='Transforms input data into 1x1x1mm^3 spacing, if True', default=True)
    parser.add_argument('--saveGif', help='Save gif image of the game', action='store_true', default=False)
    parser.add_argument('--saveVideo', help='Save video of the game', action='store_true', default=False)
    parser.add_argument('--log_dir', help='Store logs in this directory during training', default= os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-1]), type=str)
    parser.add_argument('--output_dir_name', help='Directory name for storing logs during training.', default='training_output_test', type=str)
    parser.add_argument('--log_comment', help='Suffix (current time) appended to the name of the log folder name.',default='', type=str)
    parser.add_argument('--agents', help='Number of agents detecting the landmark', type=int, default=5)
    parser.add_argument('--batch_size', help='Size of each batch, default: 100', default=100, type=int)
    parser.add_argument('--memory_size', help="""Number of transitions stored in exp replay buffer. If too much is allocated training may abruptly stop.""", default=1e5, type=int)
    parser.add_argument('--init_memory_size', help='Number of transitions stored in exp replay before training', default=3e4, type=int)
    parser.add_argument('--discount', help='Discount factor used in the Bellman equation', default=0.9, type=float)
    parser.add_argument('--lr', help='Starting learning rate', default=1e-4, type=float)
    parser.add_argument('--scheduler_gamma', help='Multiply the learning rate by this value every scheduler_step_size epochs',  default=0.05, type=float)
    parser.add_argument('--scheduler_step_size', help='Every scheduler_step_size epochs, the learning rate is multiplied by scheduler_gamma', default=10, type=int)
    parser.add_argument('--max_episodes', help='"Number of episodes to train for"', default=10, type=int)
    parser.add_argument('--steps_per_episode', help='Maximum steps per episode', default=300, type=int) 
    parser.add_argument('--target_update_freq', help='Number of epochs between each target network update', default=10, type=int)
    parser.add_argument('--save_freq', help='Saves network every save_freq steps', default=10, type=int)
    parser.add_argument('--delta', help='Amount to decreases epsilon each episode, for the epsilon-greedy policy', default=1e-4, type=float)
    parser.add_argument('--viz', help='Size of the window, None for no visualisation', default=None, type=float)
    parser.add_argument('--multiscale', help='Reduces size of voxel around the agent when it oscillates', dest='multiscale', action='store_true', default='True')
    parser.add_argument('--write', help='Saves the training logs', dest='write', action='store_true', default=True)
    parser.add_argument('--team_reward', help='Refers to adding the (potentially weighted) average reward of all agents to their individiual rewards', choices=[None, 'mean', 'attention'], default='mean')
    parser.add_argument('--attention', help='Use attention for communication channel in C-MARL', dest='attention', action='store_true', default=False)
    parser.add_argument('--train_freq', help='Number of agent steps between each training step on one mini-batch', default=10, type=int)
    parser.add_argument('--seed', help="Random seed for both training and evaluating. If none is provided, no seed will be set", type=int)
    parser.add_argument('--fixed_spawn', nargs='*', type=float, help='Starting position of the agents during rollout. Randomised if not specified.')
    
    args = parser.parse_args()

    task = 'train'
    assert args.agents > 0

    init_memory_size = min(args.init_memory_size, args.memory_size)

    if args.seed is not None:
        set_reproducible(args.seed)

    logger = Logger(args.log_dir, args.output_dir_name, args.write, args.save_freq, comment=args.log_comment)
    landmark_ids = [0 for _ in range(args.agents)]
    print('Data preparation for training in progress ...')
    environment = get_player(task=task,
                             files_list   = args.files,
                             agents       = args.agents,
                             viz          = args.viz,
                             multiscale   = args.multiscale,
                             logger       = logger,
                             landmark_ids = landmark_ids,
                             norm_spacing = args.normalised_spacing_wanted_for_proceeding)
    eval_env = None
    if args.val_files is not None:
        print('Data preparation for evaluation in progress ...')
        eval_env = get_player(task='eval',
                              files_list   = args.val_files,
                              agents       = args.agents,
                              logger       = logger,
                              landmark_ids = landmark_ids,
                              norm_spacing = args.normalised_spacing_wanted_for_proceeding)
            
    trainer = Trainer(environment,
                      eval_env            = eval_env,
                      batch_size          = args.batch_size,
                      image_size          = IMAGE_SIZE,
                      frame_history       = FRAME_HISTORY,
                      update_frequency    = args.target_update_freq,
                      replay_buffer_size  = args.memory_size,
                      init_memory_size    = init_memory_size,
                      gamma               = args.discount,
                      steps_per_episode   = args.steps_per_episode,
                      max_episodes        = args.max_episodes,
                      delta               = args.delta,
                      logger              = logger,
                      train_freq          = args.train_freq,
                      team_reward         = args.team_reward,
                      attention           = args.attention,
                      lr                  = args.lr,
                      scheduler_gamma     = args.scheduler_gamma,
                      scheduler_step_size = args.scheduler_step_size
                      ).train()
