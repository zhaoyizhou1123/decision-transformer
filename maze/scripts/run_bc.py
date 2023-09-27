# Combo style dynamics model

import __init__

import argparse
import os
import pickle
import wandb
import time

from torch.utils.tensorboard import SummaryWriter  
from maze.utils.dataset import TrajCtxDataset, TrajNextObsDataset, ObsActDataset
from maze.algos.bc.bc_trainer import TrainerConfig
# from maze.algos.decision_transformer.models.decision_transformer import DecisionTransformer
# from maze.algos.decision_transformer.training.seq_trainer import SequenceTrainer
from maze.utils.buffer import ReplayBuffer
from maze.utils.trajectory import Trajs2Dict
from maze.utils.scalar import StandardScaler
# from maze.algos.stitch_rcsl.models.dynamics_normal import EnsembleDynamicsModel
from maze.utils.logger import Logger, make_log_dirs
from maze.utils.none_or_str import none_or_str
from maze.utils.setup_logger import setup_logger
from maze.utils.setup_seed import setup_seed
from gym_dt.decision_transformer.models.mlp_bc import MLPBCModel
from gym_dt.decision_transformer.training.act_trainer import ActTrainer
import torch
import numpy as np

def run(args):
    if args.env_type == 'pointmaze':
        from create_maze_dataset import create_env_dataset
        point_maze_offline = create_env_dataset(args, use_wrapper = True)
        env = point_maze_offline.env_cls()

        s = env.reset()
        print(s)

        # Add a get_true_observation method for Env
        # def get_true_observation(obs):
        #     '''
        #     obs, obs received from pointmaze Env
        #     '''
        #     return obs['observation']
    
        # setattr(env, 'get_true_observation', get_true_observation)

        horizon = args.horizon
        obs_shape = env.observation_space.shape
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        trajs = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info

        assert len(trajs[0].observations) == horizon, f"Horizon mismatch: {len(trajs[0].observations)} and {horizon}"
        
        # Get dict type dataset, for dynamics training
        dynamics_dataset = Trajs2Dict(trajs)
    else:
        raise Exception(f"Unimplemented env type {args.env_type}")

    # horizon = env.get_horizon()
    # num_actions = env.get_num_action()
    # print(f"Num_actions = {num_actions}")

    # initialize a trainer instance and kick off training
    epochs = args.epochs

    # Set up environment
    # if args.hash:
    #     hash_method = state_hash
    # else:
    #     hash_method = None

    cur_time = time.localtime(time.time())
    format_time = f"{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}{cur_time.tm_hour:02d}{cur_time.tm_min:02d}"
    data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"
    if args.tb_path is not None:
        # Create tb log dir
        # data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"
        # if args.algo == 'dt':
        #     tb_dir = f"{args.algo}_{data_file}_ctx{args.ctx}_batch{args.batch}_goal{args.goal}_lr{args.rate}"
        if args.algo == "rcsl-mlp": # 'mlp'
            sample_method = 'sample' if args.sample else 'top'
            if args.arch == '/':
                args.arch = ''
            tb_dir = f"{args.algo}_{data_file}_ctx{args.ctx}_arch{args.arch}_{sample_method}_rep{args.repeat}_batch{args.batch}_goalmul{args.goal_mul}_lr{args.rate}"
            if args.simple_input:
                tb_dir += "_simpleinput"
        else:
            raise Exception(f"Unimplemented algorithm {args.algo}")
        tb_dir_path = os.path.join(args.tb_path,format_time,tb_dir)
        os.makedirs(tb_dir_path, exist_ok=False)
    else:
        tb_dir_path = None



    # Set up wandb
    if args.log_to_wandb:
        wandb.init(
            name=f"{data_file}_{format_time}",
            group=data_file,
            project='rcsl',
            config=args
        )
        
    train_dataset = TrajCtxDataset(trajs, ctx = 1, single_timestep = False, keep_ctx=False)

    # Use CQL logger
    from SimpleSAC.utils import WandBLogger
    from viskit.logging import logger as logger_cql, setup_logger as setup_logger_cql
    # logging_conf = WandBLogger.get_default_config(updates={"prefix":'stitch-mlp',
    #                                                        "project": 'stitch-rcsl',
    #                                                        "output_dir": './mlp_log'})
    # wandb_logger = WandBLogger(config=logging_conf, variant=vars(args))
    setup_logger_cql(
        variant=vars(args),
        exp_id=f"arch{args.arch}--mlp",
        seed=args.seed,
        base_log_dir="./rcsl-mlp_log/",
        include_exp_prefix_sub_dir=False
    )

    # num_offline = len(trajs)
    # num_rollout = len(rollout_trajs)

    # if (args.offline_ratio == 0): # rollout only
    #     train_dataset = TrajCtxDataset(rollout_trajs, 
    #                                 ctx = args.ctx, single_timestep = False)
    # else:
    #     repeat_rollout = math.ceil(num_offline / args.offline_ratio * (1-args.offline_ratio) / num_rollout)

    #     train_dataset = TrajCtxDataset(trajs + rollout_trajs * repeat_rollout, 
    #                                 ctx = args.ctx, single_timestep = False)

    # Update: use weighted sampling, setup in Trainer
        
    setup_seed(args.seed)
    env.reset(seed=args.seed)
    # from maze.algos.stitch_rcsl.models.mlp_policy import RcslPolicy
    # model = RcslPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
    #                 simple_input=args.simple_input)
    arch = args.arch.split("-")
    model = MLPBCModel(
        state_dim = obs_dim,
        act_dim = action_dim,
        hidden_size = int(arch[0]),
        n_layer = len(arch)
    )

    optim =  torch.optim.AdamW(
        model.parameters(),
        lr=args.rate
    )
    output_policy_trainer = ActTrainer(
        model,
        optim,
        batch_size = args.batch,

    )
    # output_policy_trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
    print("Begin output policy training")
    output_policy_trainer.train()

    print("Begin output policy training")
    output_policy_trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=123)

    # Overall configuration
    parser.add_argument('--maze_config_file', type=str, default='./config/maze2.json')
    parser.add_argument('--data_file', type=str, default='./dataset/maze2.dat')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    parser.add_argument('--algo', type=str, default='stitch-mlp-rolloutonly', help="rcsl-mlp, rcsl-dt or stitch-mlp, stitch-cql")
    parser.add_argument('--horizon', type=int, default=200, help="Should be consistent with dataset")
    parser.add_argument('--num_workers', type=int, default=1, help="Dataloader workers")

    # Output policy
    parser.add_argument('--ctx', type=int, default=1)
    # parser.add_argument('--pre_epochs', type=int, default=5, help='epochs to learn the output policy using offline data')
    parser.add_argument('--offline_ratio', type=float, default=0.5, help='ratio of offline data in whole dataset, only useful in stitch')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs to learn the output policy')
    parser.add_argument('--step_per_epoch', type=int, default=1000, help='number of training steps per epoch')
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    # parser.add_argument('--num_steps', type=int, default=500000)
    # parser.add_argument('--num_buffers', type=int, default=50)
    # parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch', type=int, default=256, help='final output policy training batch')
    parser.add_argument('--final_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable. Used to store output policy model" )
    parser.add_argument('--arch', type=str, default='200-200-200-200', help="Hidden layer size of output mlp model" )
    parser.add_argument('--goal_mul', type=float, default=1, help="goal = max_dataset_return * goal_mul")

    # DT output policy
    parser.add_argument('--embed_dim', type=int, default=128, help="dt token embedding dimension")
    parser.add_argument('--n_layer', type=int, default=3, help="Transformer layer")
    parser.add_argument('--n_head', type=int, default=1, help="Transformer head")    
    
    
    # parser.add_argument('--mdp_batch', type=int, default=128)
    # parser.add_argument('--mdp_epochs', type=int, default=5, help='epochs to learn the mdp model')
    # parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    
    # parser.add_argument('--rollout_data_file', type=str, default=None, help='./dataset/maze_rollout.dat')
    # parser.add_argument('--log_level', type=str, default='WARNING')
    parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of Trainer" )
    parser.add_argument('--hash', action='store_true', help="Hash states if True")
    # parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
    # parser.add_argument('--env_path', type=str, default='./env/env_rev.txt', help='Path to env description file')
    parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension, default -1 for no embedding")
    # parser.add_argument('--n_layer', type=int, default=1, help="Transformer layer")
    # parser.add_argument('--n_head', type=int, default=1, help="Transformer head")
    # parser.add_argument('--model', type=str, default='dt', help="mlp or dt")
    parser.add_argument('--r_loss_weight', type=float, default=0.5, help="[0,1], weight of r_loss" )
    parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
    parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")
    parser.add_argument('--time_depend_s',action='store_true')
    parser.add_argument('--time_depend_a',action='store_true')
    parser.add_argument('--simple_input',action='store_false', help='Only use history rtg info if true')

    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--test_rollout", action='store_true')
    args = parser.parse_args()
    print(args)

    run(args=args)