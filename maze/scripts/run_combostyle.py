# Combo style dynamics model

import __init__

import argparse
import os
import pickle
import wandb
import time

from torch.utils.tensorboard import SummaryWriter  
from maze.utils.dataset import TrajCtxDataset, TrajNextObsDataset, ObsActDataset
from maze.scripts.rollout import rollout_expand_trajs, test_rollout_combo, rollout_combo
from maze.algos.stitch_rcsl.training.trainer_dynamics import EnsembleDynamics 
from maze.algos.stitch_rcsl.training.trainer_base import TrainerConfig
from maze.algos.stitch_rcsl.training.train_mdp import BehaviorPolicyTrainer
from maze.algos.stitch_rcsl.models.mlp_policy import StochasticPolicy
from maze.utils.buffer import ReplayBuffer
from maze.utils.trajectory import Trajs2Dict
from maze.utils.scalar import StandardScaler
from maze.algos.stitch_rcsl.models.dynamics_normal import EnsembleDynamicsModel
from maze.utils.logger import Logger, make_log_dirs
from maze.utils.none_or_str import none_or_str
from maze.utils.setup_logger import setup_logger
import torch
import numpy as np

def run(args):
    if args.env_type == 'pointmaze':
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
        obs_shape = env.observation_space['observation'].shape
        obs_dim = env.observation_space['observation'].shape[0]
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

    # Get current device
    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'

    # Load rollout trajs if possible
    # if args.rollout_ckpt_path is not None and os.path.exists(args.rollout_ckpt_path):
    #     print("Rollout trajs exist, load rollout trajs")
    #     with open(args.rollout_ckpt_path, "rb") as f:
    #         rollout_trajs = pickle.load(args.rollout_ckpt_path)
    # # Load fitted model if possible
    # elif args.mdp_ckpt_dir is not None and os.path.exists(args.mdp_ckpt_dir):
    #     print("MDP model exists, load existing models")
    #     d_model_path = os.path.join(args.mdp_ckpt_dir, "dynamics.pth")
    #     b_model_path = os.path.join(args.mdp_ckpt_dir, "behavior.pth")
    #     i_model_pth = os.path.join(args.mdp_ckpt_dir, "init.pth")
    #     dynamics_model = torch.load(d_model_path, map_location=device)
    #     behavior_model = torch.load(b_model_path, map_location=device)
    #     init_model = torch.load(i_model_pth, map_location=device)
    #     dynamics_model.train(False)
    #     behavior_model.train(False)
    #     init_model.train(False)

    #     print("Rollout")
    #     rollout_trajs = rollout_expand_trajs(dynamics_model, behavior_model, init_state_model,
    #                                         device,args)
    # # Fit model, rollout
    # else:


    # create buffer
    real_buffer = ReplayBuffer(
        buffer_size=len(trajs) * horizon, # number of transitions
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device=device
    )
    real_buffer.load_dataset(dynamics_dataset)

    # log
    log_dirs = make_log_dirs(args.env_type, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "dynamics_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # Create dynamics model
    dynamics_model = EnsembleDynamicsModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=args.dynamics_hidden_dims,
        num_ensemble=args.n_ensemble,
        num_elites=args.n_elites,
        weight_decays=args.dynamics_weight_decay,
        device=device
    )
    dynamics_optim = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=args.dynamics_lr
    )

    scaler = StandardScaler()
    dynamics_trainer = EnsembleDynamics(
        dynamics_model,
        dynamics_optim,
        scaler,
    )

    # Load trained model
    if args.load_dynamics_path:
        print(f"Loading dynamics model from {args.load_dynamics_path}")
        dynamics_trainer.load(args.load_dynamics_path)
    else:
        print(f"Training dynamics model")
        dynamics_trainer.train(real_buffer.sample_all(), logger, max_epochs_since_update=5)   

    # Get behavior policy model 
    if args.behavior_type == 'mlp': # mlp model
        if args.mdp_ckpt_dir is not None and os.path.exists(args.mdp_ckpt_dir):
            print(f"MDP model exists, load existing models from {args.mdp_ckpt_dir}")
            # d_model_path = os.path.join(args.mdp_ckpt_dir, "dynamics.pth")
            b_model_path = os.path.join(args.mdp_ckpt_dir, "behavior.pth")
            # i_model_pth = os.path.join(args.mdp_ckpt_dir, "init.pth")
            # dynamics_model = torch.load(d_model_path, map_location=device)
            behavior_model = torch.load(b_model_path, map_location=device)
            # init_model = torch.load(i_model_pth, map_location=device)
            # dynamics_model.train(False)
            behavior_model.train(False)
            # init_model.train(False)
        else: # Train behavior policy model
            if args.mdp_ckpt_dir is None:
                ckpt_path = os.path.join('./checkpoint',format_time)
            else:
                ckpt_path = args.mdp_ckpt_dir
            os.makedirs(ckpt_path,exist_ok=True)

            model_dataset = TrajNextObsDataset(trajs)
            print(f"Train behavior policy model")

            conf = TrainerConfig(max_epochs=args.mdp_epochs, batch_size=args.behavior_batch, learning_rate=args.rate,
                                num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                                ckpt_path = ckpt_path, tb_log = tb_dir_path, r_loss_weight = args.r_loss_weight,
                                weight_decay = 0.1, log_to_wandb = args.log_to_wandb)
            
            behavior_model = StochasticPolicy(obs_dim, action_dim, arch = args.b_arch, n_support = args.n_support)


            behavior_trainer = BehaviorPolicyTrainer(behavior_model, model_dataset, conf)
            behavior_trainer.train()
    elif args.behavior_type == 'diffusion': # diffusion policy
        # config for both training and logger setup
        conf = TrainerConfig(obs_dim = obs_dim,
                             act_dim = action_dim,
                             spectral_norm = False,
                             num_epochs = args.behavior_epoch,
                             num_diffusion_iters = args.num_diffusion_iters,
                             batch_size = args.behavior_batch,
                             algo = args.behavior_type,
                             env_id = args.env_type,
                             expr_name = 'default',
                             seed = args.diffusion_seed,
                             load = args.load_diffusion)
        
        state_action_dataset = ObsActDataset(trajs)
        # Configure logger
        diffusion_logger = setup_logger(conf)

        from maze.algos.stitch_rcsl.training.trainer_diffusion import DiffusionBC
        diffusion_policy = DiffusionBC(conf, state_action_dataset, diffusion_logger)

        if not args.load_diffusion:
            print(f"Train diffusion behavior policy")
            diffusion_policy.train()
            diffusion_policy.save_checkpoint()
        else:
            print(f"Load diffusion behavior policy")
            diffusion_policy.load_checkpoint()

    else:
        raise Exception(f"Unimplemented behavior policy type {args.behavior_type}")

    # if args.test_rollout:
    #     # Test rollout
    #     test_rollout_combo(args, dynamics_trainer, real_buffer, behavior_model_raw=behavior_model, based_true_state=False, init_true_state=True)
    # else:
    #     rollout_trajs = rollout_combo(args, dynamics_trainer, behavior_model, real_buffer)

    #     train_dataset = TrajCtxDataset(trajs + rollout_trajs, ctx = args.ctx, single_timestep = False)

    #     if args.algo == 'rcsl-mlp':
    #         # num_action = env.get_num_action()
    #         # print(f"num_action={num_action}")
    #         # model = MlpPolicy(1, num_action, args.ctx, horizon, args.repeat, args.arch, args.n_embd)
    #         from maze.algos.rcsl.models.mlp_policy import MlpPolicy

    #         model = MlpPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
    #                         simple_input=args.simple_input)
    #         import maze.algos.rcsl.trainer_mlp as trainer_mlp
    #         goal = train_dataset.get_max_return() * args.goal_mul
    #         tconf = trainer_mlp.TrainerConfig(max_epochs=epochs, batch_size=args.batch, learning_rate=args.rate,
    #                     lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
    #                     desired_rtg=goal, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
    #                     ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb)
    #         output_policy_trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
    #     elif args.algo == 'stitch':
    #         from maze.algos.stitch_rcsl.models.mlp_policy import RcslPolicy
    #         model = RcslPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
    #                         simple_input=args.simple_input)
            
    #         goal = train_dataset.get_max_return() * args.goal_mul

    #         import maze.algos.stitch_rcsl.training.trainer_mlp as trainer_mlp
    #         tconf = trainer_mlp.TrainerConfig(max_epochs=epochs, batch_size=args.batch, learning_rate=args.rate,
    #                     lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
    #                     desired_rtg=goal, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
    #                     ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb)
    #         output_policy_trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
    #     else:
    #         raise(Exception(f"Unimplemented model {args.algo}!"))

    #     print("Begin output policy training")
    #     output_policy_trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ctx', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5, help='epochs to learn the output policy')
    parser.add_argument('--mdp_epochs', type=int, default=5, help='epochs to learn the mdp model')
    parser.add_argument('--rollout_epochs', type=int, default=10, help="Number of epochs to rollout the policy")
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    # parser.add_argument('--num_steps', type=int, default=500000)
    # parser.add_argument('--num_buffers', type=int, default=50)
    # parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--mdp_batch', type=int, default=128)
    parser.add_argument('--maze_config_file', type=str, default='./config/maze2.json')
    
    # parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--data_file', type=str, default='./dataset/maze2.dat')
    parser.add_argument('--rollout_data_file', type=str, default=None, help='./dataset/maze_rollout.dat')
    parser.add_argument('--log_level', type=str, default='WARNING')
    parser.add_argument('--goal_mul', type=float, default=1, help="goal = max_dataset_return * goal_mul")
    parser.add_argument('--horizon', type=int, default=250, help="Should be consistent with dataset")
    parser.add_argument('--ckpt_prefix', type=none_or_str, default=None, help="Used to store output policy model" )
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="file path, used to load/store rollout trajs" )
    parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of Trainer" )
    parser.add_argument('--hash', action='store_true', help="Hash states if True")
    parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    # parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
    # parser.add_argument('--env_path', type=str, default='./env/env_rev.txt', help='Path to env description file')
    parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension, default -1 for no embedding")
    # parser.add_argument('--n_layer', type=int, default=1, help="Transformer layer")
    # parser.add_argument('--n_head', type=int, default=1, help="Transformer head")
    # parser.add_argument('--model', type=str, default='dt', help="mlp or dt")
    parser.add_argument('--algo', type=str, default='stitch', help="rcsl-mlp or stitch")
    parser.add_argument('--b_arch', type=str, default='128', help="Hidden layer size of behavior model" )
    parser.add_argument('--n_support', type=int, default=10, help="Number of supporting action of policy" )
    parser.add_argument('--n_support_init', type=int, default=5, help="Number of supporting initial states" )
    parser.add_argument('--d_arch', type=str, default='128', help="Hidden layer size of dynamics model" )
    parser.add_argument('--r_loss_weight', type=float, default=0.5, help="[0,1], weight of r_loss" )
    parser.add_argument('--arch', type=str, default='256', help="Hidden layer size of output mlp model" )
    parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
    parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")
    parser.add_argument('--time_depend_s',action='store_true')
    parser.add_argument('--time_depend_a',action='store_true')
    parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    parser.add_argument('--simple_input',action='store_false', help='Only use history rtg info if true')
    parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--render', action='store_true')

    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--seed", type=int, default=1)

    # Dynamics model
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--rollout-batch-size", type=int, default=50000)
    parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=none_or_str, default=None)

    # Behavior policy
    parser.add_argument('--behavior_type', type=str, default='diffusion', help='mlp or diffusion')
    parser.add_argument('--mdp_ckpt_dir', type=str, default='./checkpoint/test_dynamics', help="dir path, used to load/store mdp model" )
    parser.add_argument("--behavior_epoch", type=int, default=5)
    parser.add_argument("--num_diffusion_iters", type=int, default=100, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion', action="store_true", help="Load diffusion policy model if true")
    parser.add_argument('--diffusion_seed', type=int, default=0, help="Distinguish runs for diffusion policy")

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--test_rollout", action='store_true')
    args = parser.parse_args()
    print(args)

    run(args=args)