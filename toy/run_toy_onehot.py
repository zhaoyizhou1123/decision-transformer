# Dataset: Should use torch.utils.data.Dataset
# make deterministic
# from mingpt.utils import set_seed
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import math
# import torch
import argparse
from env.bandit_dataset import BanditReturnDataset, read_data
from toy_nn.model_att import DTConfig, SimpleDT
from toy_nn.trainer import TrainerConfig, Trainer
from env.no_best_RTG import BanditEnvOneHot as Env
import json
import logging

# from __future__ import print_function, division
# import os
# import torch
# import pandas as pd
# # from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms, utils

parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5)
# parser.add_argument('--model_type', type=str, default='reward_conditioned')
# parser.add_argument('--num_steps', type=int, default=500000)
# parser.add_argument('--num_buffers', type=int, default=50)
# parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=1)
# 
# parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_file', type=str, default='./dataset/toy.csv')
# parser.add_argument('--log_level', type=str, default='WARNING')
parser.add_argument('--goal', type=int, default=5, help="The desired RTG")
parser.add_argument('--horizon', type=int, default=5, help="Should be consistent with dataset")
parser.add_argument('--ckpt_prefix', type=str, default=None )
parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of Trainer" )
parser.add_argument('--n_embd', type=int, default=40, help="token embedding dimension")
parser.add_argument('--weight_decay', type=float, default=0.1, help="weight decay for Trainer optimizer" )
parser.add_argument('--init_att', type=str, default=None, help="initial value of attention params, will convert to float" )
parser.add_argument('--freeze_att', type=bool, default=False, help="Freeze attention params if True" )
args = parser.parse_args()

# print args
print(args)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# Set MDP
env = Env(args.horizon)

# Get space_dim and action_dim
state_dim, action_dim = env.get_dims()

# Get the dataset
states, actions, rtgs, timesteps = read_data(args.data_file, args.horizon, action_dim)
# print(f"Read data actions: {actions.shape}")
dataset = BanditReturnDataset(states, args.context_length*3, actions, rtgs, timesteps)

# Parse init_att
if args.init_att is None:
    init_att = None
else:
    init_att = float(args.init_att)

# model configuration 
mconf = DTConfig(state_dim = state_dim,
                 n_act = action_dim, 
                 n_embd = args.n_embd,
                 horizon = args.horizon, 
                 ctx = args.context_length,
                 init_att = init_att, 
                 freeze_att = args.freeze_att)

model = SimpleDT(mconf)

# trainer configuration
tconf = TrainerConfig(batch_size = args.batch_size, 
                      num_workers = 4,
                      grad_norm_clip = 1.0,
                      max_epochs = args.epochs,
                      desired_rtg = args.goal, 
                      ckpt_prefix = args.ckpt_prefix, 
                      env = env,
                      eval_repeat = 10,
                      ctx_length = args.context_length, 
                      horizon = args.horizon,
                      lr = args.rate, 
                      weight_decay = args.weight_decay)
trainer = Trainer(model, dataset, tconf)

trainer.train()
