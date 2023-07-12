'''
Learn the dynamics and behavior policy from dataset
'''
import __init__

import argparse
from maze.algos.stitch_rcsl.training.train_mdp import TrainerConfig, MdpTrainer
from maze.algos.stitch_rcsl.models.mdp_model import DynamicsModel
from maze.algos.stitch_rcsl.models.mlp_policy import BehaviorPolicy, StochasticPolicy
from maze.utils.dataset import TrajNextObsDataset
import pickle
import time
import os
import wandb

def learn(trajs, state_dim, action_dim, tb_dir_path, ckpt_path, args):
    '''
    trajs: list(Trajectory), got from dataset
    '''
    # with open(args.data_fle, 'rb') as file:
    #     trajs, _, _, _ = pickle.load(file) # Dataset may not be trajs, might contain other infos

    model_dataset = TrajNextObsDataset(trajs)

    conf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch, learning_rate=args.rate,
                        num_workers=1, horizon=args.horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                        ckpt_path = ckpt_path, tb_log = tb_dir_path, 
                        weight_decay = 0.1, log_to_wandb = args.log_to_wandb)
    
    dynamics_model = DynamicsModel(state_dim, action_dim, arch = args.d_arch)
    # behavior_model = BehaviorPolicy(state_dim, action_dim, arch = args.b_arch)
    behavior_model = StochasticPolicy(state_dim, action_dim, arch = args.b_arch, n_support = args.n_support)

    mdp_trainer = MdpTrainer(dynamics_model, behavior_model, model_dataset, conf)
    mdp_trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=100)
    # parser.add_argument('--num_steps', type=int, default=500000)
    # parser.add_argument('--num_buffers', type=int, default=50)
    # parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    # parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--data_file', type=str, default='./dataset/maze.dat')
    parser.add_argument('--horizon', type=int, default=250, help="Should be consistent with dataset")
    parser.add_argument('--ckpt_root', type=str, default=None, help="./checkpoint/, path to checkpoint root dir" )
    parser.add_argument('--ckpt_name', type=str, default=None, help="Name of dir, to prompt the data" )
    parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of Trainer" )
    parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/" )
    parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension, default -1 for no embedding")
    parser.add_argument('--algo', type=str, default='rcsl-mlp', help="rcsl-mlp or ?")
    parser.add_argument('--b_arch', type=str, default='128', help="Hidden layer size of behavior model" )
    parser.add_argument('--n_support', type=int, default=2, help="Number of supporting action of policy" )
    parser.add_argument('--d_arch', type=str, default='128', help="Hidden layer size of dynamics model" )
    parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
    parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")
    parser.add_argument('--time_depend_s',action='store_true')
    parser.add_argument('--time_depend_a',action='store_true')
    parser.add_argument('--simple_input',action='store_false', help='Only use history rtg info if true')
    parser.add_argument('--log_to_wandb',action='store_false', help='Set up wandb')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    args = parser.parse_args()
    print(args)

    if args.env_type == 'pointmaze': # Create env and dataset
        from create_maze_dataset import create_env_dataset
        point_maze_offline = create_env_dataset(args)
        env = point_maze_offline.env_cls()

        # Add a get_true_observation method for Env
        def get_true_observation(obs):
            '''
            obs, obs received from pointmaze Env
            '''
            return obs['observation']
    
        setattr(env, 'get_true_observation', get_true_observation)

        horizon = args.horizon
        obs_dim = env.observation_space['observation'].shape[0]
        action_dim = env.action_space.shape[0]
        trajs = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info
        assert len(trajs[0].observations) == horizon, f"Horizon mismatch: {len(trajs[0].observations)} and {horizon}"
    else:
        raise Exception(f"Unimplemented env type {args.env_type}")
    
    cur_time = time.localtime(time.time())
    format_time = f"{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}{cur_time.tm_hour:02d}{cur_time.tm_min:02d}"

    # Create tb log dir
    data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"

    if args.tb_path is not None:
        if args.algo == "rcsl-mlp": # 'mlp'
            sample_method = 'sample' if args.sample else 'top'
            if args.d_arch == '/':
                args.d_arch = ''
            if args.b_arch == '/':
                args.b_arch = ''
            tb_dir = f"mdp_{args.algo}_{data_file}_archdb{args.d_arch}_{args.b_arch}_{sample_method}_rep{args.repeat}_batch{args.batch}_lr{args.rate}"
            if args.simple_input:
                tb_dir += "_simpleinput"
        else:
            raise Exception(f"Unimplemented algorithm {args.algo}")
        tb_dir_path = os.path.join(args.tb_path,format_time,tb_dir)
        os.makedirs(tb_dir_path, exist_ok=False)
    else:
        tb_dir_path = None # No tb

    if args.log_to_wandb:
        wandb.init(
            name=f"{data_file}_{format_time}_mdp",
            group=data_file,
            project='rcsl',
            config=args
        )

    if args.ckpt_root is not None:
        if args.ckpt_name is None:
            ckpt_path = os.path.join(args.ckpt_root,format_time)
        else:
            ckpt_path = os.path.join(args.ckpt_root,f"{args.ckpt_name}")
        os.makedirs(ckpt_path,exist_ok=True)

    learn(trajs, obs_dim, action_dim, tb_dir_path, ckpt_path, args=args)