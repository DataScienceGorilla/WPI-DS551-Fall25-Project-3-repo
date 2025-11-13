#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, in_channels=4, num_actions=4):
        """
        Parameters:
        -----------
        in_channels: number of channel of input.
            i.e The number of most recent frames stacked together, here we use 4 frames, which means each state in Breakout is composed of 4 frames.
        num_actions: number of action-value to output, one-to-one correspondence to action in game.

        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(DQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # This architecture is straight from the deepmind paper
        # Input shape: (N, 4, 84, 84)
        
        # Conv Layer 1
        # (N, 4, 84, 84) -> (N, 32, 20, 20)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        
        # Conv Layer 2
        # (N, 32, 20, 20) -> (N, 64, 9, 9)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        
        # Conv Layer 3
        # (N, 64, 9, 9) -> (N, 64, 7, 7)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        
        # Fully Connected Layer 1
        # The flattened output size from conv3 is 64 * 7 * 7 = 3136
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=512)
        
        # Fully Connected Layer 2 (Output)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
        

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        
        Note: The input x is assumed to be a tensor of shape (N, C, H, W) 
              e.g., (batch_size, 4, 84, 84)
              and is already pre-processed (e.g., normalized to [0, 1]).
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # Pass input through conv layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the output for the fully-connected layers
        # x.view(-1, ...) is a standard way to flatten / debug to reshape
        # -1 infers the batch size
        x = x.reshape(-1, 64 * 7 * 7)
        
        # Pass through the first fully-connected layer with ReLU
        x = F.relu(self.fc1(x))
        
        # Pass through the output layer (no activation function)
        # We want raw Q-values as output
        x = self.fc2(x)
        
        ###########################
        return x