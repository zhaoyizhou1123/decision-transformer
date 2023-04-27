# Dataset: Should use torch.utils.data.Dataset
import csv
import logging
# make deterministic
# from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# import math
from torch.utils.data import Dataset
from mingpt.model_toy import GPT, GPTConfig
from mingpt.trainer_toy import Trainer, TrainerConfig
# from mingpt.utils import sample
# from collections import deque
# import random
import torch
# import pickle
# import blosc
import argparse
# from create_dataset import create_dataset

# from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
# parser.add_argument('--num_steps', type=int, default=500000)
# parser.add_argument('--num_buffers', type=int, default=50)
# parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=1)
# 
# parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_file', type=str, default='./dataset/toy.csv')
parser.add_argument('--log_level', type=str, default='WARNING')
parser.add_argument('--goal', type=int, default=5, help="The desired RTG")
parser.add_argument('--horizon', type=int, default=5, help="Should be consistent with dataset")
parser.add_argument('--ckpt_prefix', type=str, default=None )
args = parser.parse_args()
# print(args)

# Read dataset
data_frame = pd.read_csv(args.data_file)
# print(data_frame)
base_idxs = np.arange(0,4*args.horizon,4)
timesteps = data_frame.iloc[:,np.append(base_idxs,4*args.horizon)]

# full batch
states = data_frame.iloc[:,base_idxs+1]
actions = data_frame.iloc[:,base_idxs+2]
rewards = data_frame.iloc[:,base_idxs+3]
# timesteps = np.asarray(timesteps)
timesteps=np.asarray(timesteps)
states=np.asarray(states)
actions=np.asarray(actions)
rewards = np.asarray(rewards)

# create rtgs array
rtgs = np.zeros_like(rewards) # n*args.horizon, n is trajectory number
assert rewards.shape[1] == args.horizon, "Horizon configuration error!"
rtgs[:,args.horizon-1] = rewards[:,args.horizon-1]
for i in range(args.horizon-2,-1,-1): # backward
    rtgs[:, i] = rtgs[:,i+1] + rewards[:,i]

# print(timesteps)
# print(states)
# print(actions)
# print(rewards)
# print(rtgs)

class BanditReturnDataset(Dataset):
    '''Son of the pytorch Dataset class'''

    def __init__(self, states, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size # args.context_length*3
        self.vocab_size = 2 # Specifies number of actions. The actions are represented s {0,1,2,...}
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        self.data = states #obss
        self.actions = actions
        self.done_idxs = done_idxs # What is this for?
        self.rtgs = rtgs
        self.timesteps = timesteps
        self._trajectory_num = states.shape[0] # number of trajectories in dataset
        self._horizon = states.shape[1]

        # number of trajectories should match
        assert self._trajectory_num == actions.shape[0] 
        assert self._trajectory_num == rtgs.shape[0] 
        assert self._trajectory_num == timesteps.shape[0] 
    
    def __len__(self):
        return self._trajectory_num*(self._horizon - self.block_size // 3 + 1)
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        '''
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - states: Tensor of size [ctx_length, state_space_size]
        - actions: Tensor of size [ctx_length, 1]
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: Tensor of size [1,1], the starting time of the RTG trajectory slice
        '''
        # block_size = self.block_size // 3  # "//" means [5/3], here approx context length
        # done_idx = idx + block_size
        # for i in self.done_idxs:
        #     if i > idx: # first done_idx greater than idx
        #         done_idx = min(int(i), done_idx)
        #         break
        # idx = done_idx - block_size
        # states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.

        ctx = self.block_size // 3 # context length
        data_num_per_trajectory = self._horizon - self.block_size // 3 + 1  # number of data one trajectory provides
        trajectory_idx = idx // data_num_per_trajectory # which trajectory to read, row
        res_idx = idx - trajectory_idx * data_num_per_trajectory # column index to read
        
        assert res_idx + ctx <= args.horizon, idx
        assert trajectory_idx < self._trajectory_num, idx
        states_slice = torch.tensor(self.data[trajectory_idx, res_idx : res_idx + ctx]).unsqueeze(1)
        actions_slice = torch.tensor(self.actions[trajectory_idx, res_idx : res_idx + ctx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs_slice = torch.tensor(self.rtgs[trajectory_idx, res_idx : res_idx + ctx], dtype=torch.float32).unsqueeze(1)
        timesteps_slice = torch.tensor(self.timesteps[trajectory_idx, res_idx : res_idx + 1], dtype=torch.int64).unsqueeze(1)
        
        # print(f"Size of output: {states_slice.shape}, {actions_slice.shape}, {rtgs_slice.shape}, {timesteps_slice.shape}")

        return states_slice, actions_slice, rtgs_slice, timesteps_slice
    
    def getitem(self, idx):
        states_slice, actions_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
        return states_slice, actions_slice, rtgs_slice, timesteps_slice

# obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# dataset = BanditReturnDataset(states, 3*args.context_length, actions, None, rtgs, timesteps)
# states_slice, actions_slice, rtgs_slice, timesteps_slice=dataset.getitem(4)
# print(f"states:{states_slice}")
# print(f"actions:{actions_slice}")
# print(f"rtgs:{rtgs_slice}")
# print(f"timesteps:{timesteps_slice}")

# print("Begin generating train_dataset")
train_dataset = BanditReturnDataset(states, args.context_length*3, actions, None, rtgs, timesteps)
print(train_dataset.len())
# print("Finish generation")

# print("Begin GPT configuartion.")
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=1, n_head=5, n_embd=10, model_type=args.model_type, max_timestep=args.horizon)
# print("End GPT config, begin model generation")
model = GPT(mconf)
# print("End model generation")

# initialize a trainer instance and kick off training
epochs = args.epochs

# print("Begin Trainer configuartion")
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-3,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*train_dataset.len()*args.context_length*3,
                      num_workers=4, model_type=args.model_type, max_timestep=args.horizon, horizon=args.horizon, 
                      desired_rtg=args.goal, ckpt_prefix = args.ckpt_prefix)
# print("End trainer configuration, begin trainer generation")
trainer = Trainer(model, train_dataset, None, tconf)

# print("End trainer generation. Begin training.")
trainer.train()