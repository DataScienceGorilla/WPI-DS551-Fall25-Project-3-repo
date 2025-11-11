#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
import os
import sys
import time  # <-- ADD THIS IMPORT

import torch
import torch.nn.functional as F
import torch.optim as optim

# We import the Agent class as our base class
from agent import Agent 
# We import the DQN model
from dqn_model import DQN
"""
you can import any package and define any extra function as you need
"""

torch.manual_seed(595)
np.random.seed(595)
random.seed(595)


class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize everything you need here.
        For example: 
            parameters for neural network 
            initialize Q net and target Q net
            parameters for replay buffer
            parameters for q-learning; decaying epsilon-greedy
            ...
        """

        super(Agent_DQN,self).__init__(env)
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # --- Hyperparameters ---
        # These are common hyperparameters for DQN.
        # You can also pass these in via the 'args' object.
        self.buffer_size = 30000  # Size of the replay buffer
        self.batch_size = 32       # Batch size for training
        self.gamma = 0.99          # Discount factor
        self.learning_rate = 0.0001 # Learning rate for the optimizer
        self.target_update_freq = 1000 # How often to update the target network (in steps)
        
        # Epsilon-greedy parameters for exploration
        self.eps_start = 1.0
        self.eps_end = 0.1
        self.eps_decay_frames = 300000 # Over how many frames to decay epsilon
        
        # Calculate the decay step size
        self.epsilon = self.eps_start
        self.eps_decay_step = (self.eps_start - self.eps_end) / self.eps_decay_frames

        # --- Training Loop Hyperparameters ---
        self.n_episodes = 10000  # Total number of episodes to train for
        self.save_path = "dqn_model.pth" # Path to save the model
        
        # --- Device Setup ---
        # Set device to GPU (cuda) if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # --- Network Setup ---
        # Get action space size from the environment
        self.num_actions = env.action_space.n
        
        # Initialize the Policy Network (the one we train)
        # We move it to the specified device.
        self.policy_net = DQN(num_actions=self.num_actions).to(self.device)
        
        # Initialize the Target Network (for stable target Q-values)
        # We also move it to the device.
        self.target_net = DQN(num_actions=self.num_actions).to(self.device)
        
        # Copy the weights from the policy_net to the target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set the target_net to evaluation mode (it's not being trained directly)
        self.target_net.eval()

        # --- Optimizer ---
        # We use the Adam optimizer (a common choice)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        # --- Replay Buffer ---
        # We use a deque as a fixed-size buffer
        self.memory = deque(maxlen=self.buffer_size)
        
        # --- Training State ---
        self.steps_done = 0 # Counter for total steps, used for epsilon decay and target updates

        
        if args.test_dqn:
            # If we are in test mode, load a pre-trained model
            print('loading trained model')
            ###########################
            # YOUR IMPLEMENTATION HERE #
            # Load the saved model weights. Make sure the path is correct.
            # 'map_location' ensures it loads correctly even if trained on GPU and testing on CPU.
            self.policy_net.load_state_dict(torch.load("dqn_model.pth", map_location=self.device))
            # Set the policy net to evaluation mode
            self.policy_net.eval()
            

    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary.
        If no parameters need to be initialized, you can leave it as blank.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # For a vanilla DQN, we don't need to reset anything special at the
        # start of a new game (our epsilon is based on total steps).
        ###########################
        pass
    
    
    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
        Return:
            action: int
                the predicted action from trained model
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # This is our Epsilon-Greedy action selection
        
        if test:
            # During testing, we want to be mostly greedy (exploit).
            # We still add a tiny 1% exploration to avoid getting stuck.
            epsilon = 0.01 
        else:
            # During training, we use the decaying epsilon
            epsilon = self.epsilon

        sample = random.random()
        
        if sample > epsilon:
            # --- EXPLOIT: Choose the best action from the Q-network ---
            
            # 1. Convert observation to a PyTorch Tensor
            # Input is (H, W, C) = (84, 84, 4)
            # PyTorch needs (N, C, H, W) = (1, 4, 84, 84)
            # .permute(2, 0, 1) changes (84, 84, 4) -> (4, 84, 84)
            # .unsqueeze(0) adds the batch dimension -> (1, 4, 84, 84)
            obs = torch.from_numpy(observation).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # 2. Set model to evaluation mode (important for batchnorm, dropout)
            self.policy_net.eval()
            
            # 3. Get Q-values from the network
            with torch.no_grad(): # We don't need to calculate gradients here
                q_values = self.policy_net(obs)
            
            # 4. Set model back to train mode if we are training
            if not test:
                self.policy_net.train()

            # 5. Select the action with the highest Q-value
            # .max(1) finds the max value along dimension 1 (the actions)
            # [1] gets the indices (the actions themselves)
            # .item() converts the tensor to a single Python integer
            action = q_values.max(1)[1].item()
            
        else:
            # --- EXPLORE: Choose a random action ---
            action = random.randrange(self.num_actions)
            
        ###########################
        return action
    
    def push(self, state, action, reward, next_state, done):
        """ 
        Push new data (a transition) to the replay buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        # We store the transition tuple in our deque
        self.memory.append((state, action, reward, next_state, done))
        ###########################
        
        
    def replay_buffer(self):
        """ 
        Sample a batch of transitions from the buffer.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # 1. Sample a random batch of transitions from memory
        batch = random.sample(self.memory, self.batch_size)
        
        # 2. Unzip the batch into separate lists
        # This is a cool Python trick: zip(*batch) transposes the list of tuples
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 3. Convert all transitions to PyTorch tensors
        
        # Convert states and next_states
        # np.stack turns the list of (84, 84, 4) arrays into (32, 84, 84, 4)
        # .permute changes it to (32, 4, 84, 84) for our network
        states = torch.from_numpy(np.stack(states)).permute(0, 3, 1, 2).float().to(self.device)
        next_states = torch.from_numpy(np.stack(next_states)).permute(0, 3, 1, 2).float().to(self.device)
        
        # Convert actions, rewards, and dones
        # .unsqueeze(1) adds a dimension to make them (32, 1) for calculations
        actions = torch.tensor(actions, device=self.device, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float).unsqueeze(1)
        # Convert dones (True/False) to 1.0/0.0
        dones = torch.tensor(dones, device=self.device, dtype=torch.float).unsqueeze(1)
        
        ###########################
        return states, actions, rewards, next_states, dones
        

    def train(self):
        """
        Implement your training algorithm here
        This function is called ONCE by main.py and contains the entire training loop.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        print("Starting training...")
        start_time = time.time()
        rewards_deque = deque(maxlen=100) # For tracking avg reward
        
        for i_episode in range(1, self.n_episodes + 1):
            # Reset environment and get initial state
            # self.env is provided by the base Agent class
            state = self.env.reset() 
            episode_reward = 0
            
            while True:
                # 1. Agent chooses an action
                action = self.make_action(state, test=False)
                
                # 2. Take action in the environment
                # The env is already wrapped by the 'Environment' class in main.py
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # 3. Push the experience to the agent's memory
                self.push(state, action, reward, next_state, done)
                
                # 4. Tell the agent to train (it will if buffer is ready)
                self._optimize_model() # <-- CALL THE RENAMED FUNCTION
                
                # 5. Update state and total reward
                state = next_state
                episode_reward += reward
                
                # 6. Check if the episode is over
                if done or truncated:
                    break
            
            # --- End of Episode ---
            rewards_deque.append(episode_reward)
            avg_reward = np.mean(rewards_deque)
            
            if i_episode % 100 == 0:
                elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f'Episode {i_episode}/{self.n_episodes} | Avg Reward (Last 100): {avg_reward:.2f} | Epsilon: {self.epsilon:.4f} | Time: {elapsed}')
                
                # Save the model
                torch.save(self.policy_net.state_dict(), self.save_path)

        print(f"Training finished. Model saved to {self.save_path}")
        
        ###########################


    def _optimize_model(self):
        """
        Implement the optimization step.
        This function is called once per frame (or step) in the main training loop.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        
        # 1. Don't train until we have enough samples in the buffer
        if len(self.memory) < self.batch_size:
            return

        # 2. Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer()

        # --- Calculate Q(s, a) ---
        # 3. Get the Q-values for all states in the batch from the policy_net
        #    Q_values will have shape (32, num_actions)
        Q_all = self.policy_net(states)
        
        # 4. Select the Q-value for the action that was *actually taken*
        #    .gather(1, actions) picks the value from Q_all at the index specified by 'actions'
        Q_expected = Q_all.gather(1, actions)

        # --- Calculate Y_target = r + γ * max_a' Q_target(s', a') ---
        # 5. Get the max Q-value for the *next_state* from the *target_net*
        with torch.no_grad(): # We don't need gradients for the target net
            # .max(1)[0] gets the max Q-value (not the action index)
            # .unsqueeze(1) makes its shape (32, 1) to match Q_expected
            Q_next_max = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # 6. Calculate the target Q-value
        #    If 'done' is 1.0 (True), the second term becomes 0.
        Q_targets = rewards + (self.gamma * Q_next_max * (1 - dones))

        # --- Perform the optimization step ---
        # 7. Calculate the loss (Huber loss is common for DQNs)
        loss = F.smooth_l1_loss(Q_expected, Q_targets)
        
        # 8. Backpropagation
        self.optimizer.zero_grad() # Clear old gradients
        loss.backward()            # Calculate new gradients
        
        # 9. Gradient Clipping (prevents exploding gradients)
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()       # Update the policy_net's weights

        # --- Update epsilon and target network ---
        
        # 10. Decay epsilon
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_decay_step)
        
        # 11. Update the target network
        # This is a "hard update" every 'target_update_freq' steps
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        ###########################