# Dataset: Should use torch.utils.data.Dataset
import torch
import argparse
from env.bandit_dataset import BanditRewardDataset, read_data_reward
from cql.model_cql import FullyConnectedQFunction
# from env.no_best_RTG import BanditEnv as Env
from env.bandit_env import BanditEnv
from env.time_var_env import TimeVarEnv, TimeVarBanditEnv
import logging
import os
from env.utils import OneHotHash, sample
from env.linearq_dataset import DictQDataset, traj_rtg_datasets
from env.bandit_env import BanditEnv
from env.linearq import Linearq
from env.time_var_env import TimeVarEnv, TimeVarBanditEnv
from offlinerlkit.utils.logger import Logger, make_log_dirs
import time
import json
import random
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--env_param", type = int, default = 160)
parser.add_argument("--seed", type = int, default = 0)
# parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epochs', type=int, default=50)
# parser.add_argument('--model_type', type=str, default='reward_conditioned')
# parser.add_argument('--num_steps', type=int, default=500000)
# parser.add_argument('--num_buffers', type=int, default=50)
# parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--logger_dir', default="logs/linearq/cql")
# 
# parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_file', type=str, default='./dataset/toy.csv')
# parser.add_argument('--log_level', type=str, default='WARNING')
# parser.add_argument('--horizon', type=int, default=20, help="Should be consistent with dataset")
parser.add_argument('--ckpt_prefix', type=str, default=None )
parser.add_argument('--rate', type=float, default=1e-3, help="learning rate of Trainer" )
parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension")
parser.add_argument('--weight_decay', type=float, default=0.1, help="weight decay for Trainer optimizer" )
parser.add_argument('--arch', type=str, default='160', help="Hidden layer size of Q-function" )
parser.add_argument('--tradeoff_coef', type=float, default=0, help="alpha in CQL" )
parser.add_argument('--env_path', type=str, default='./env/env_rev.txt', help='Path to env description file')
parser.add_argument('--tb_path', type=str, default="./logs/linearq/cql", help="Folder to tensorboard logs" )
parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
parser.add_argument('--scale', type=float, default=1.0, help="Scale the reward")
parser.add_argument('--time_depend_s',action='store_true')
parser.add_argument('--time_depend_a',action='store_true')
parser.add_argument('--env_type', type=str, default='bandit', help='bandit or timevar or timevar_bandit')
parser.add_argument('--hash_a',type=str, default=None, help='onehot')
parser.add_argument('--shuffle',action='store_false')
parser.add_argument('--tabular',action='store_true')
parser.add_argument('--train_mode', type=str, default='dqn', help='dqn or old')
parser.add_argument('--dqn_upd_period', type=int, default='10', help='dqn target network update period')

args = parser.parse_args()

# print args
args.task="linearq"
args.algo_name="qlearning"
print(args)

# logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         level=logging.INFO,
# )

# Get action hash function. The temp_env is used to get action space
# temp_env = Env(args.env_path, sample=sample, state_hash=None, action_hash=None, 
#                time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)
# hash = OneHotHash(temp_env.get_num_action()).hash

# Set MDP, no hash
# env = Env(args.env_path, sample=sample, state_hash=None, action_hash=hash, 
        #   time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

env = Linearq(args.env_param, reward_mul=1)

horizon = env.horizon
num_actions = 2
# print(f"Num_actions = {num_actions}")


# states, actions, rtgs, timesteps = read_data(args.data_file, horizon)
# train_dataset = BanditReturnDataset(states, args.context_length*3, actions, rtgs, timesteps, single_timestep=True)
dataset, _, max_offline_return = traj_rtg_datasets(env)
train_dataset = DictQDataset(dataset, horizon = horizon)
# s,a,r,t,ns =  train_dataset[0]
# print(s,a,r,t,ns)
print(len(train_dataset))
# print("Finish generation")

# time_var_env method
# horizon = args.horizon
# assert horizon % 2 == 0, f"Horizon {horizon} must be even!"
# num_actions = horizon // 2

if args.hash_a is None:
    hash = None
elif args.hash_a == 'onehot':
    hash = OneHotHash(num_actions).hash
    env._action_hash = hash
else:
    raise Exception(f"Invalid args.hash_a !")



# Get the dataset. Actions are hashed in BanditRewardDataset
# states, true_actions, rewards, timesteps = read_data_reward(args.data_file, horizon)
# # print(f"Read data actions: {actions.shape}")
# dataset = BanditRewardDataset(states, true_actions, rewards, timesteps, state_hash=None, action_hash=hash, time_order=not args.shuffle)
# # dataset = BanditRewardDataset(states, true_actions, rewards, timesteps, state_hash=None, action_hash=None)

# Create tb log dir
# data_name = args.data_file[10:-4] # Like "env_rev", args.env_path form "./env/xxx.csv"
# if args.arch == '/':
#     args.arch = ''
# tb_dir = f"{data_name}_scale{args.scale}_arch{args.arch}_repeat{args.repeat}_alpha{args.tradeoff_coef}_embd{args.n_embd}_batch{args.batch_size}_lr{args.rate}"

# if args.hash_a == 'one_hot':
#     tb_dir += f"_{args.hash_a}"
# if not args.shuffle:
#     tb_dir += f"_noshuffle"
# if args.tabular:
#     tb_dir += f"_tabular"
# if args.train_mode == 'dqn':
#     tb_dir += f"_dqn_period{args.dqn_upd_period}"

cur_time = time.localtime(time.time())
format_time = f"{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}{cur_time.tm_hour:02d}{cur_time.tm_min:02d}{cur_time.tm_sec:02d}"
# tb_dir_path = os.path.join(args.tb_path,format_time)
# os.makedirs(tb_dir_path, exist_ok=False)
# with open(os.path.join(tb_dir_path, "hyper_param.json"), "w") as f:
#     json.dump(vars(args), f, indent = 4)
# os.makedirs(tb_dir_path, exist_ok=False)

# Remember to change the observed action dim according to the hashing method
# observed_action_dim = env.get_num_action()

rcsl_log_dirs = make_log_dirs(args.task, args.algo_name, f"param{args.env_param}-arch{args.arch}", vars(args))
# key: output file name, value: output handler type
rcsl_output_config = {
    "consoleout_backup": "stdout",
    "policy_training_progress": "csv",
    # "dynamics_training_progress": "csv",
    "tb": "tensorboard"
}
rcsl_logger = Logger(rcsl_log_dirs, rcsl_output_config)
rcsl_logger.log_hyperparameters(vars(args))

if args.train_mode == 'old':
    from cql.trainer_cql import TrainerConfig, Trainer

    if args.hash_a == 'onehot':
        action_dim = num_actions
    else:
        action_dim = 1

    model = FullyConnectedQFunction(observation_dim=1,
                                    action_dim=action_dim, 
                                    horizon=horizon, 
                                    arch=args.arch,
                                    token_repeat=args.repeat, 
                                    embd_dim=args.n_embd,
                                    is_tabular=args.tabular)

    # action_space = torch.Tensor([[0],[1]])



    # trainer configuration
    tconf = TrainerConfig(batch_size = args.batch_size, 
                        num_workers = 1,
                        grad_norm_clip = 1.0,
                        max_epochs = args.epochs, 
                        ckpt_prefix = args.ckpt_prefix, 
                        env = env,
                        eval_repeat = 1,
                        horizon = horizon,
                        lr = args.rate, 
                        weight_decay = args.weight_decay,
                        tradeoff_coef = args.tradeoff_coef,
                        tb_log = None, 
                        r_scale = args.scale,
                        shuffle = args.shuffle,
                        )
    trainer = Trainer(model, train_dataset, tconf)
elif args.train_mode == 'dqn':
    from cql.trainer_dqn import TrainerConfig, Trainer
    from cql.model_cql import DqnConfig

    logger_dir = os.path.join(args.logger_dir, f"param{args.env_param}-arch{args.arch}", format_time)
    os.makedirs(logger_dir, exist_ok = True)
    with open(os.path.join(logger_dir, "hyper_param.json"), "w") as f:
        json.dump(vars(args), f, indent = 4,)
    # logger_path = os.path.join(logger_dir, "result.log")
    # logging.basicConfig(filename=logger_path, filemode = 'w', level=logging.INFO)
    # print(f"Logging to {logger_path}")
    model_config = DqnConfig(1, num_actions, args.arch)
    trainer_config = TrainerConfig(batch_size = args.batch_size, 
                        num_workers = 0,
                        grad_norm_clip = 1.0,
                        max_epochs = args.epochs, 
                        ckpt_prefix = args.ckpt_prefix, 
                        env = env,
                        eval_repeat = 1,
                        horizon = horizon,
                        lr = args.rate, 
                        weight_decay = args.weight_decay,
                        tradeoff_coef = args.tradeoff_coef,
                        tb_log = None, 
                        r_scale = args.scale,
                        shuffle = args.shuffle,
                        target_upd_period = args.dqn_upd_period,
                        # logger_path = logger_path
                        logger = rcsl_logger
                        )
    
    trainer = Trainer(model_config, train_dataset, trainer_config)
else:
    raise Exception(f"Unknown train_mode {args.train_mode}")


trainer.train()
