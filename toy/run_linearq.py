# TODO: modify to run new environments

# Dataset: Should use torch.utils.data.Dataset
import csv
import logging
# make deterministic
# from mingpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from mingpt.model_toy import GPT, GPTConfig
from mingpt.mlp import MlpPolicy
import mingpt.trainer_toy as trainer_toy
import mingpt.trainer_mlp as trainer_mlp
from mingpt.utils import state_hash
import torch
import argparse
# from create_dataset import create_dataset
import os
# from env.utils import sample
from env.linearq_dataset import DictDataset, traj_rtg_datasets
from env.bandit_env import BanditEnv
from env.linearq import Linearq
from env.time_var_env import TimeVarEnv, TimeVarBanditEnv
import time
import json
import random

parser = argparse.ArgumentParser()
# env config (linearq)
parser.add_argument("--env_param", type = int, default = 10)
parser.add_argument("--seed", type = int, default = 0)

# parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=2)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
# parser.add_argument('--num_steps', type=int, default=500000)
# parser.add_argument('--num_buffers', type=int, default=50)
# parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=8000)
# 
# parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_file', type=str, default='./dataset/toy.csv')
parser.add_argument('--log_level', type=str, default='WARNING')
parser.add_argument('--goal_mul', type=float, default=1., help="The desired RTG")
parser.add_argument('--horizon', type=int, default=20, help="Should be consistent with dataset")
parser.add_argument('--ckpt_prefix', type=str, default=None )
parser.add_argument('--rate', type=float, default=1e-4, help="learning rate of Trainer" )
parser.add_argument('--hash', action='store_true', help="Hash states if True")
parser.add_argument('--tb_path', type=str, default="./logs/linearq/mlp", help="Folder to tensorboard logs" )
parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
parser.add_argument('--env_path', type=str, default='./env/env_rev.txt', help='Path to env description file')
parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension, default -1 for no embedding")
parser.add_argument('--n_layer', type=int, default=1, help="Transformer layer")
parser.add_argument('--n_head', type=int, default=1, help="Transformer head")
parser.add_argument('--model', type=str, default='mlp', help="mlp or dt")
parser.add_argument('--arch', type=str, default='2', help="Hidden layer size of mlp model" )
parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")
parser.add_argument('--time_depend_s',action='store_true')
parser.add_argument('--time_depend_a',action='store_true')
parser.add_argument('--env_type', type=str, default='bandit', help='bandit or timevar or timevar_bandit or linearq')
parser.add_argument('--simple_input',action='store_false', help='Only use history rtg info if true')
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

env = Linearq(args.env_param, reward_mul=1)

horizon = env.horizon
num_action = 2
# print(f"Num_actions = {num_actions}")


# states, actions, rtgs, timesteps = read_data(args.data_file, horizon)
# train_dataset = BanditReturnDataset(states, args.context_length*3, actions, rtgs, timesteps, single_timestep=True)
dataset, _, max_offline_return = traj_rtg_datasets(env)
train_dataset = DictDataset(dataset, horizon = horizon)
print(len(train_dataset))
# print("Finish generation")

# Set models
if args.model == 'dt':
    # Set GPT parameters
    n_layer = args.n_layer
    n_head = args.n_head
    n_embd = args.n_embd
    print(f"GPTConfig: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

    # print("Begin GPT configuartion.")
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                    n_layer=n_layer, n_head=n_head, n_embd=n_embd, model_type=args.model_type, max_timestep=horizon)
    # print("End GPT config, begin model generation")
    model = GPT(mconf)
    # print("End model generation")
elif args.model == 'mlp':
    # num_action = env.get_num_action()
    # print(f"num_action={num_action}")
    # model = MlpPolicy(1, num_action, args.context_length, horizon, args.repeat, args.arch, args.n_embd)
    model = MlpPolicy(1, num_action, args.context_length, horizon, args.repeat, args.arch, args.n_embd,
                      simple_input=args.simple_input)
else:
    raise(Exception(f"Unimplemented model {args.model}!"))

# initialize a trainer instance and kick off training
epochs = args.epochs

# Set up environment
if args.hash:
    hash_method = state_hash
else:
    hash_method = None

# Create tb log dir
# data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"
# if args.model == 'dt':
#     tb_dir = f"{args.model}_{data_file}_ctx{args.context_length}_batch{args.batch_size}_goal{args.goal}_lr{args.rate}"
# else: # 'mlp'
#     sample_method = 'sample' if args.sample else 'top'
#     if args.arch == '/':
#         args.arch = ''
#     tb_dir = f"{args.model}_{data_file}_ctx{args.context_length}_arch{args.arch}_{sample_method}_rep{args.repeat}_embd{args.n_embd}_batch{args.batch_size}_goal{args.goal}_lr{args.rate}"
#     if args.simple_input:
#         tb_dir += "_simpleinput"
cur_time = time.localtime(time.time())
format_time = f"{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}{cur_time.tm_hour:02d}{cur_time.tm_min:02d}"
tb_dir_path = os.path.join(args.tb_path,format_time)
os.makedirs(tb_dir_path, exist_ok=False)
with open(os.path.join(tb_dir_path, "hyper_param.json"), "w") as f:
    json.dump(vars(args), f, indent = 4)

# print("Begin Trainer configuartion")
if args.model == 'dt':
    tconf = trainer_toy.TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=args.rate,
                lr_decay=True, warmup_tokens=512*20, final_tokens=2*train_dataset.len()*args.context_length*3,
                num_workers=1, model_type=args.model_type, max_timestep=horizon, horizon=horizon, 
                desired_rtg=max_offline_return * args.goal_mul, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
                ctx = args.context_length)
# print("End trainer configuration, begin trainer generation")
    trainer = trainer_toy.Trainer(model, train_dataset, None, tconf)
elif args.model == 'mlp':
    tconf = trainer_mlp.TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=args.rate,
                lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                desired_rtg=max_offline_return * args.goal_mul, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
                ctx = args.context_length, sample = args.sample, num_action = num_action)
    trainer = trainer_mlp.Trainer(model, train_dataset, tconf)

# print("End trainer generation. Begin training.")
trainer.train()