'''
Create the Dataset Class from csv file
'''

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BanditReturnDataset(Dataset):
    '''Son of the pytorch Dataset class'''

    def __init__(self, states, block_size, actions, rtgs, timesteps):        
        self.block_size = block_size # args.context_length*3
        self.vocab_size = 2 # Specifies number of actions. The actions are represented s {0,1,2,...}
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        self.data = states #obss
        self.actions = actions # np.array (num_trajectories, horizon)
        # self.done_idxs = done_idxs # What is this for?
        self.rtgs = rtgs
        self.timesteps = timesteps # (trajectory_num,horizon+1)
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
        - actions: Tensor of size [ctx_length, action_dim], here action is converted to one-hot representation
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: Tensor of size [ctx_length]
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
        
        assert res_idx + ctx <= self._horizon, idx
        assert trajectory_idx < self._trajectory_num, idx
        states_slice = self.data[trajectory_idx, res_idx : res_idx + ctx, :]
        actions_slice = self.actions[trajectory_idx, res_idx : res_idx + ctx, :]
        rtgs_slice = self.rtgs[trajectory_idx, res_idx : res_idx + ctx, :]
        timesteps_slice = self.timesteps[trajectory_idx, res_idx : res_idx + ctx]
        
        # print(f"Size of output: {states_slice.shape}, {actions_slice.shape}, {rtgs_slice.shape}, {timesteps_slice.shape}")
        # print(f"Dataset actions_slice: {actions_slice.shape}")
        return states_slice, actions_slice, rtgs_slice, timesteps_slice
    
    def getitem(self, idx):
        states_slice, actions_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
        return states_slice, actions_slice, rtgs_slice, timesteps_slice
    

def onehot_convert(action, action_dim):
    '''
    action: int in [0,action_dim-1] \n
    action_dim: int \n
    Return: Tensor (action_dim), one-hot representation
    '''
    action = int(action)
    assert 0<=action and action <= action_dim-1
    onehot_action = torch.zeros(action_dim)
    onehot_action[action] = 1
    return onehot_action



def read_data(data_file, horizon, action_dim):
    '''
    data_file, str, path to data file (Bugs of relative paths?) \n
    action_dim, int, dimension of actions, or number of possible actions
    Return values:
    - states, torch.Tensor (num_trajectories, horizon, state_dim). 
    - onehot_actions, (num_trajectories, horizon, action_dim). Here action_dim=2
    - rtgs, (num_trajectories, horizon, 1)
    - timesteps: (num_trajectories, horizon+1). Starting timestep is adjusted to 0
    '''
    # Read dataset
    data_frame = pd.read_csv(data_file)
    # print(data_frame)
    base_idxs = np.arange(0,4*horizon,4)
    timesteps = data_frame.iloc[:,np.append(base_idxs,4*horizon)]

    # full batch
    states = data_frame.iloc[:,base_idxs+1]
    actions = data_frame.iloc[:,base_idxs+2]
    rewards = data_frame.iloc[:,base_idxs+3]
    # timesteps = np.asarray(timesteps)
    timesteps = torch.Tensor(np.asarray(timesteps)) # (num_trajectory, horizon+1)
    # Adjust starting timestep to 0
    start_time = timesteps[0,0] # the starting time step
    timesteps = timesteps - start_time
    states = torch.Tensor(np.asarray(states)).unsqueeze(-1) # (num_trajectory, horizon, 1)
    actions = np.asarray(actions)
    rewards = np.asarray(rewards)

    # create rtgs array
    rtgs = np.zeros((rewards.shape[0], rewards.shape[1])) # (n, args.horizon), n is trajectory number
    assert rewards.shape[1] == horizon, "Horizon configuration error!"
    rtgs[:,horizon-1] = rewards[:,horizon-1]
    for i in range(horizon-2,-1,-1): # backward
        rtgs[:, i] = rtgs[:,i+1] + rewards[:,i]
    rtgs = torch.Tensor(rtgs).unsqueeze(-1) # (num_trajectory, horizon, 1)

    # convert actions into onehot representation
    onehot_actions = torch.zeros((actions.shape[0],actions.shape[1], action_dim))
    for i in range(onehot_actions.shape[0]):
        for j in range(onehot_actions.shape[1]):
            onehot_actions[i,j,:] = onehot_convert(actions[i,j], action_dim)

    return states, onehot_actions, rtgs, timesteps