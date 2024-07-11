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
import torch.nn as nn

class Network3D(nn.Module):

    def __init__(self, agents, frame_history, number_actions, xavier=True):
        super(Network3D, self).__init__()

        self.agents = agents
        self.frame_history = frame_history
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.conv0 = nn.Conv3d(
            in_channels=frame_history,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool0 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu0 = nn.PReLU().to(self.device)
        self.conv1 = nn.Conv3d(
            in_channels=32,
            out_channels=32,
            kernel_size=(5, 5, 5),
            padding=1).to(
            self.device)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu1 = nn.PReLU().to(self.device)
        self.conv2 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=(4, 4, 4),
            padding=1).to(
            self.device)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)).to(self.device)
        self.prelu2 = nn.PReLU().to(self.device)
        self.conv3 = nn.Conv3d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3, 3),
            padding=0).to(
            self.device)
        self.prelu3 = nn.PReLU().to(self.device)

        self.fc1 = nn.ModuleList(
            [nn.Linear(in_features=512, out_features=256).to(
                self.device) for _ in range(self.agents)])
        self.prelu4 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc2 = nn.ModuleList(
            [nn.Linear(in_features=256, out_features=128).to(
                self.device) for _ in range(self.agents)])
        self.prelu5 = nn.ModuleList(
            [nn.PReLU().to(self.device) for _ in range(self.agents)])
        self.fc3 = nn.ModuleList(
            [nn.Linear(in_features=128, out_features=number_actions).to(
                self.device) for _ in range(self.agents)])

        if xavier:
            for module in self.modules():
                if type(module) in [nn.Conv3d, nn.Linear]:
                    torch.nn.init.xavier_uniform(module.weight)

    def forward(self, input):
        """
        Input is a tensor of size:  (batch_size, agents, frame_history, *image_size)
        Output is a tensor of size: (batch_size, agents, number_actions)
        """
        input = input.to(self.device) / 255.0
        output = []
        for i in range(self.agents):
            # Shared layers
            x = input[:, i]
            x = self.conv0(x)
            x = self.prelu0(x)
            x = self.maxpool0(x)
            x = self.conv1(x)
            x = self.prelu1(x)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.prelu2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.prelu3(x)
            x = x.reshape(-1, 512)
            # Individual layers
            x = self.fc1[i](x)
            x = self.prelu4[i](x)
            x = self.fc2[i](x)
            x = self.prelu5[i](x)
            x = self.fc3[i](x)
            output.append(x)
        output = torch.stack(output, dim=1)
        return output.cpu()



class DQN:
    # The class initialisation function.
    def __init__(
            self,
            agents,
            frame_history,
            logger,
            number_actions=6,
            collective_rewards=False,
            attention=False,
            lr=1e-3,
            scheduler_gamma=0.9,
            scheduler_step_size=100):
        self.agents = agents
        self.number_actions = number_actions
        self.frame_history = frame_history
        self.logger = logger
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.logger.log(f"Using {self.device}")
        # Create a Q-network, which predicts the q-value for a particular state
        self.q_network = Network3D(
            agents,
            frame_history,
            number_actions).to(
            self.device)
        self.target_network = Network3D(
            agents, frame_history, number_actions).to(
            self.device)
        if collective_rewards == "attention":
            self.q_network.rew_att = nn.Parameter(torch.randn(agents, agents))
            self.target_network.rew_att = nn.Parameter(torch.randn(agents, agents))
        self.copy_to_target_network()
        # Freezes target network
        self.target_network.train(False)
        for p in self.target_network.parameters():
            p.requires_grad = False
        # Define the optimiser which is used when updating the Q-network. The
        # learning rate determines how big each gradient step is during
        # backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimiser, step_size=scheduler_step_size, gamma=scheduler_gamma)
        self.collective_rewards = collective_rewards

    def copy_to_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, name="dqn.pt", forced=False):
        self.logger.save_model(self.q_network.state_dict(), name, forced)

    # Function that is called whenever we want to train the Q-network. 
    # Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transitions, discount_factor):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transitions, discount_factor)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transitions, discount_factor):
        '''
        Transitions are tuple of shape
        (states, actions, rewards, next_states, dones)
        '''
        curr_state = torch.tensor(transitions[0])
        next_state = torch.tensor(transitions[3])
        terminal = torch.tensor(transitions[4]).type(torch.int)

        rewards = torch.clamp(
            torch.tensor(
                transitions[2], dtype=torch.float32), -1, 1)
        # Collective rewards here refers to adding the (potentially weighted) average reward of all agents
        if self.collective_rewards == "mean":
            rewards += torch.mean(rewards, axis=1).unsqueeze(1).repeat(1, rewards.shape[1])
        elif self.collective_rewards == "attention":
            rewards = rewards + torch.matmul(rewards, nn.Softmax(dim=0)(self.q_network.rew_att))

        y = self.target_network.forward(next_state)
        # dim (batch_size, agents, number_actions)
        y = y.view(-1, self.agents, self.number_actions)
        # Get the maximum prediction for the next state from the target network
        max_target_net = y.max(-1)[0]

        # dim (batch_size, agents, number_actions)
        network_prediction = self.q_network.forward(curr_state).view(
            -1, self.agents, self.number_actions)
        isNotOver = (torch.ones(*terminal.shape) - terminal)
        # Bellman equation
        batch_labels_tensor = rewards + isNotOver * \
            (discount_factor * max_target_net.detach())

        actions = torch.tensor(transitions[1], dtype=torch.long).unsqueeze(-1)
        y_pred  = torch.gather(network_prediction, -1, actions).squeeze()

        return torch.nn.SmoothL1Loss()(batch_labels_tensor.flatten(), y_pred.flatten())
