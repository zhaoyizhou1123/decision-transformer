import csv
import logging
# make deterministic
from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from mingpt.model_atari import GPT, GPTConfig
from mingpt.trainer_atari import Trainer, TrainerConfig
from mingpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
from create_dataset import create_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=30)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--num_steps', type=int, default=500000)
parser.add_argument('--num_buffers', type=int, default=50)
parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=128)
# 
parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_dir_prefix', type=str, default='./dqn_replay/')
parser.add_argument('--log_level', type=str, default='WARNING')
args = parser.parse_args()
print(args)

set_seed(args.seed)

class StateActionReturnDataset(Dataset):
    '''Son of the pytorch Dataset class'''

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):        
        self.block_size = block_size # args.context_length*3
        self.vocab_size = max(actions) + 1
        self.data = data #obss
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps # order: 0,1,2, ...
    
    def __len__(self):
        return len(self.data) - self.block_size # I think should be self.block_size // 3

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
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1) # the starting idx

        return states, actions, rtgs, timesteps # starting timesteps are not trained

obss, actions, returns, done_idxs, rtgs, timesteps = create_dataset(args.num_buffers, args.num_steps, args.game, args.data_dir_prefix, args.trajectories_per_buffer)

# print("Finish dataset creation.")

# Get logging level
if args.log_level == 'DEBUG':
    logging_level = logging.DEBUG
elif args.log_level == 'INFO':
    logging_level = logging.INFO
elif args.log_level == 'WARNING':
    logging_level = logging.WARNING
elif args.log_level == 'ERROR':
    logging_level = logging.ERROR
elif args.log_level == 'CRITICAL':
    logging_level = logging.CRITICAL
else:
    raise Exception("Unkown logging level")

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging_level,
)

# print("Begin generating train_dataset")
train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
# print("Finish generation")

# print("Begin GPT configuartion.")
mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
# print("End GPT config, begin model generation")
model = GPT(mconf)
# print("End model generation")

# initialize a trainer instance and kick off training
epochs = args.epochs

# print("Begin Trainer configuartion")
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=4, seed=args.seed, model_type=args.model_type, game=args.game, max_timestep=max(timesteps))
# print("End trainer configuration, begin trainer generation")
trainer = Trainer(model, train_dataset, None, tconf)

# print("End trainer generation. Begin training.")
trainer.train()
