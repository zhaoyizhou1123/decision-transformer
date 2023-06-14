# Dataset: Should use torch.utils.data.Dataset
import torch
import argparse
from env.bandit_dataset import BanditRewardDataset, read_data_reward
from cql.model_cql import FullyConnectedQFunction
from cql.trainer_cql import TrainerConfig, Trainer
# from env.no_best_RTG import BanditEnv as Env
from env.bandit_env import BanditEnv
from env.time_var_env import TimeVarEnv
import logging
import os
from env.utils import OneHotHash, sample


parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=123)
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
parser.add_argument('--horizon', type=int, default=20, help="Should be consistent with dataset")
parser.add_argument('--ckpt_prefix', type=str, default=None )
parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of Trainer" )
parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension")
parser.add_argument('--weight_decay', type=float, default=0.1, help="weight decay for Trainer optimizer" )
parser.add_argument('--arch', type=str, default='', help="Hidden layer size of Q-function" )
parser.add_argument('--tradeoff_coef', type=float, default=1, help="alpha in CQL" )
parser.add_argument('--env_path', type=str, default='./env/env_rev.txt', help='Path to env description file')
parser.add_argument('--tb_path', type=str, default="./logs/cql", help="Folder to tensorboard logs" )
parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
parser.add_argument('--scale', type=float, default=1.0, help="Scale the reward")
parser.add_argument('--time_depend_s',action='store_true')
parser.add_argument('--time_depend_a',action='store_true')
parser.add_argument('--time_var',action='store_true')
args = parser.parse_args()

# print args
print(args)

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# Get action hash function. The temp_env is used to get action space
# temp_env = Env(args.env_path, sample=sample, state_hash=None, action_hash=None, 
#                time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)
# hash = OneHotHash(temp_env.get_num_action()).hash

# Set MDP, no hash
# env = Env(args.env_path, sample=sample, state_hash=None, action_hash=hash, 
        #   time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)

if not args.time_var:
    env = BanditEnv(args.env_path, sample=sample, state_hash=None, action_hash=None, 
                    time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)
    horizon = env.get_horizon()
else:
    horizon = args.horizon
    assert horizon % 2 == 0, f"Horizon {horizon} must be even!"
    num_actions = horizon // 2
    env = TimeVarEnv(horizon, num_actions)

# time_var_env method
# horizon = args.horizon
# assert horizon % 2 == 0, f"Horizon {horizon} must be even!"
# num_actions = horizon // 2
# hash = OneHotHash(num_actions).hash
# env = Env(horizon, num_actions, action_hash=hash)

# Get the dataset. Actions are hashed in BanditRewardDataset
states, true_actions, rewards, timesteps = read_data_reward(args.data_file, horizon)
# print(f"Read data actions: {actions.shape}")
# dataset = BanditRewardDataset(states, true_actions, rewards, timesteps, state_hash=None, action_hash=hash)
dataset = BanditRewardDataset(states, true_actions, rewards, timesteps, state_hash=None, action_hash=None)

# Remember to change the observed action dim according to the hashing method
observed_action_dim = env.get_num_action()
model = FullyConnectedQFunction(observation_dim=1,
                                action_dim=1, 
                                horizon=horizon, 
                                arch=args.arch,
                                token_repeat=args.repeat, 
                                embd_dim=args.n_embd)

# action_space = torch.Tensor([[0],[1]])

# Create tb log dir
data_name = args.data_file[10:-4] # Like "env_rev", args.env_path form "./env/xxx.csv"
if args.arch == '/':
    args.arch = ''
tb_dir = f"{data_name}_scale{args.scale}_arch{args.arch}_repeat{args.repeat}_alpha{args.tradeoff_coef}_embd{args.n_embd}_batch{args.batch_size}_lr{args.rate}"
tb_dir_path = os.path.join(args.tb_path, args.tb_suffix, tb_dir)
os.makedirs(tb_dir_path, exist_ok=False)

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
                      tb_log = tb_dir_path, 
                      r_scale = args.scale)
trainer = Trainer(model, dataset, tconf)

trainer.train()
