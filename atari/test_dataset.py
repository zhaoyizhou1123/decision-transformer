# Test the meaning of the dataset

# import csv
# import logging
# make deterministic
# from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
# import math
from torch.utils.data import Dataset
# from mingpt.model_atari import GPT, GPTConfig
# from mingpt.trainer_atari import Trainer, TrainerConfig
# from mingpt.utils import sample
# from collections import deque
# import random
# import torch
# import pickle
# import blosc
# import argparse
# from fixed_replay_buffer import FixedReplayBuffer
from create_dataset import create_dataset
# from run_dt_atari import StateActionReturnDataset


# def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
#     # -- load data from memory (make more efficient)
#     obss = []
#     actions = []
#     returns = [0]
#     done_idxs = []
#     stepwise_returns = []

#     transitions_per_buffer = np.zeros(50, dtype=int)
#     num_trajectories = 0
#     # while len(obss) < num_steps:
#     #     buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
#     #     i = transitions_per_buffer[buffer_num]
#     #     # print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
#     frb = FixedReplayBuffer(
#         data_dir=data_dir_prefix + game + '/5/replay_logs',
#         # replay_suffix=buffer_num,
#         replay_suffix=1, # from 1 to 50
#         observation_shape=(84, 84),
#         stack_size=4,
#         update_horizon=1,
#         gamma=0.99,
#         observation_dtype=np.uint8,
#         batch_size=32,
#         replay_capacity=100000)
    
#     # print(f"data_dir={data_dir_prefix + game + '/1/replay_logs'}")
#     # print(frb._loaded_buffers)
#     if frb._loaded_buffers:
#         # done = False
#         # curr_num_transitions = len(obss)
#         # trajectories_to_load = trajectories_per_buffer
#         # while not done:

#         # batches = frb.sample_transition_batch(batch_size=1, indices=[i])
#         batches = frb.sample_transition_batch(batch_size=100000, indices=range(100000))
#         print(f"type(batches)={type(batches)}")
#         states, ac, ret, next_states, next_action, next_reward, terminal, indices = batches
#         print(f"Shape of states is {states.shape}")
#         print(f"Shape of ac is {ac.shape}")
#         print(f"Shape of ret is {ret.shape}")
#         # print(f"Shape of states is {states.shape}")
#         states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
#         # print(f"states={states}")
#         # print(f"ac={ac}")
#         # print(f"ret={ret}")
#         for idx in range(len(indices)):
#             if ret[idx] > 0:
#                 print(f"ret[{idx}]={ret[idx]}")

#     #         obss += [states]
#     #         actions += [ac[0]]
#     #         stepwise_returns += [ret[0]]
#     #         if terminal[0]:
#     #             done_idxs += [len(obss)]
#     #             returns += [0]
#     #             if trajectories_to_load == 0:
#     #                 done = True
#     #             else:
#     #                 trajectories_to_load -= 1
#     #         returns[-1] += ret[0]
#     #         i += 1
#     #         if i >= 100000:
#     #         # if i>=10:
#     #             obss = obss[:curr_num_transitions]
#     #             actions = actions[:curr_num_transitions]
#     #             stepwise_returns = stepwise_returns[:curr_num_transitions]
#     #             returns[-1] = 0
#     #             i = transitions_per_buffer[buffer_num]
#     #             done = True
#     #     num_trajectories += (trajectories_per_buffer - trajectories_to_load)
#     #     transitions_per_buffer[buffer_num] = i
#     # # print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

#     # actions = np.array(actions)
#     # returns = np.array(returns)
#     # stepwise_returns = np.array(stepwise_returns)
#     # done_idxs = np.array(done_idxs)

#     # # -- create reward-to-go dataset
#     # start_index = 0
#     # rtg = np.zeros_like(stepwise_returns)
#     # for i in done_idxs:
#     #     i = int(i)
#     #     curr_traj_returns = stepwise_returns[start_index:i]
#     #     for j in range(i-1, start_index-1, -1): # start from i-1
#     #         rtg_j = curr_traj_returns[j-start_index:i-start_index]
#     #         rtg[j] = sum(rtg_j)
#     #     start_index = i
#     # print('max rtg is %d' % max(rtg))

#     # # -- create timestep dataset
#     # start_index = 0
#     # timesteps = np.zeros(len(actions)+1, dtype=int)
#     # for i in done_idxs:
#     #     i = int(i)
#     #     timesteps[start_index:i+1] = np.arange(i+1 - start_index)
#     #     start_index = i+1
#     # print('max timestep is %d' % max(timesteps))

#     # return obss, actions, returns, done_idxs, rtg, timesteps

class StateActionReturnDataset(Dataset):
    '''Son of the pytorch Dataset class'''

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size # args.context_length*3
        self.vocab_size = max(actions) + 1
        self.data = data #obss
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def len(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps
    
    def getitem(self, idx):
        '''Never trained policies for timestep 1,2,...,ctx_length-1'''
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx) 
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(1, 1, 'Qbert', './dqn_replay/', 1)
print(f"Type of obss is {type(obss)}, length={len(obss)}, element shape is {obss[0].shape}")
# print(f"obss={obss}")
# print(f"Type of actions is {type(actions)}, length={len(actions)}, element type is {type(actions[0])}")
# print(f"actions={actions}")
# print(f"Type of returns is {type(returns)}, length={len(returns)}, element type is {type(returns)}")
# print(f"returns={returns}")
print(f"Type of done_idxs is {type(done_idxs)}, length={len(done_idxs)}, element type is {type(done_idxs)}")
print(f"done_idxs={done_idxs}")
# print(f"Type of rtg is {type(rtg)}, length={len(rtg)}, element type is {type(rtg[0])}")
# print(f"rtg={rtg}")
# print(f"Type of timesteps is {type(timesteps)}, length={len(timesteps)}, element type is {type(timesteps[0])}")
# print(f"timesteps={timesteps}")

train_dataset = StateActionReturnDataset(obss, 4*3, actions, done_idxs, rtgs, timesteps)
print(f"Length of dataset is {train_dataset.len()}")
states, actions, rtgs, timesteps=train_dataset.getitem(1072)
print(f"Shape of states is {states.shape}")
print(f"Shape of actions is {actions.shape}")
print(f"Shape of rtgs is {rtgs.shape}, value is {rtgs}")
print(f"Shape of timesteps is {timesteps.shape}, value is {timesteps}")
# print(f"Shape of states is {states.shape}")