# Combo style dynamics model
import argparse
import os
import pickle
import wandb
import time
import roboverse
import torch
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter

  
from maze.utils.dataset import TrajCtxFloatLengthDataset, TrajCtxDataset
# from maze.scripts.rollout import rollout_expand_trajs, test_rollout_combo, rollout_combo, test_rollout_diffusion, rollout_diffusion
# from maze.algos.stitch_rcsl.training.trainer_dynamics import EnsembleDynamics 
from maze.algos.stitch_rcsl.training.trainer_base import TrainerConfig
# from maze.algos.stitch_rcsl.training.train_mdp import BehaviorPolicyTrainer
# from maze.algos.stitch_rcsl.models.mlp_policy import StochasticPolicy
# from maze.algos.rcsl.training.trainer_base import TrainerConfig
from maze.algos.decision_transformer.models.decision_transformer import DecisionTransformer
from maze.algos.decision_transformer.training.seq_trainer import SequenceTrainer
from maze.utils.buffer import ReplayBuffer
from maze.utils.trajectory import Trajs2Dict
from maze.utils.scalar import StandardScaler
# from maze.algos.stitch_rcsl.models.dynamics_normal import EnsembleDynamicsModel
from maze.utils.logger import Logger, make_log_dirs
from maze.utils.none_or_str import none_or_str
from maze.utils.setup_logger import setup_logger
from maze.utils.setup_seed import setup_seed
from offlinerlkit.utils.pickplace_utils import SimpleObsWrapper, get_pickplace_dataset_dt


def run(args):
    # seed
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
    # torch.backends.cudnn.deterministic = True
    # env.reset(seed = args.seed)

    # create env and dataset
    if args.task == 'pickplace':
        env = roboverse.make('Widow250PickTray-v0')
        env = SimpleObsWrapper(env)
        obs_space = env.observation_space
        args.obs_shape = obs_space.shape
        obs_dim = np.prod(args.obs_shape)
        args.action_shape = env.action_space.shape
        action_dim = np.prod(args.action_shape)

        # offline_dataset, init_obss_dataset = get_pickplace_dataset(args.data_dir, task_weight=args.task_weight)
        trajs = get_pickplace_dataset_dt(args.data_dir)
        # args.max_action = env.action_space.high[0]
        # print(args.action_dim, type(args.action_dim

        horizon = args.horizon

        env.reset(seed=args.seed)
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
        
    train_dataset = TrajCtxFloatLengthDataset(trajs, ctx = args.ctx, single_timestep = False)

    if args.algo == 'rcsl-mlp-old':
        # num_action = env.get_num_action()
        # print(f"num_action={num_action}")
        # model = MlpPolicy(1, num_action, args.ctx, horizon, args.repeat, args.arch, args.n_embd)
        from maze.algos.rcsl.models.mlp_policy import MlpPolicy

        model = MlpPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
                        simple_input=args.simple_input)
        import maze.algos.rcsl.trainer_mlp as trainer_mlp
        goal = train_dataset.get_max_return() * args.goal_mul
        
        os.makedirs(args.final_ckpt_path, exist_ok=True)
        tconf = trainer_mlp.TrainerConfig(max_epochs=epochs, batch_size=args.batch, learning_rate=args.rate,
                    lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                    desired_rtg=goal, ckpt_path = args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
                    ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb, debug=args.debug)
        output_policy_trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
    elif args.algo == 'rcsl-mlp':
        '''
        Upd: mix offline with rollout according to args.offline_ratio
        '''
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
        from maze.algos.stitch_rcsl.models.mlp_policy import RcslPolicy
        model = RcslPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
                        simple_input=args.simple_input)
        

        # Calculate weight
        num_offline_data = len(trajs)
        # num_rollout_data = len(rollout_trajs)
        # num_total_data = num_offline_data + num_rollout_data
        # offline_weight = num_rollout_data / num_total_data # reciprocal coefficient
        # rollout_weight = num_offline_data / num_total_data

        # weights = [offline_weight] * num_offline_data + [rollout_weight] * num_rollout_data

        # offline_dataset = TrajCtxWeightedDataset(trajs, [offline_weight] * num_offline_data, ctx = args.ctx, single_timestep = False)
        train_dataset = TrajCtxDataset(trajs, ctx = args.ctx, single_timestep = False)

        # train_dataset = TrajCtxWeightedDataset(
        #     trajs + rollout_trajs,
        #     weights)

        # Get max return
        traj_rets = [traj.returns[0] for traj in trajs]
        goal = max(traj_rets) * args.goal_mul
        # goal = train_dataset.get_max_return() * args.goal_mul

        import maze.algos.stitch_rcsl.training.trainer_mlp as trainer_mlp
        tconf = trainer_mlp.TrainerConfig(max_epochs=args.epochs, batch_size=args.batch, learning_rate=args.rate,
                    lr_decay=True, num_workers=args.num_workers, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                    desired_rtg=goal, ckpt_path = args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
                    ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb, logger = logger_cql,
                    debug=args.debug, seed = args.seed)
        output_policy_trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
        print("Begin output policy training")
        output_policy_trainer.train()
    # elif args.algo == 'stitch':
    #     from maze.algos.stitch_rcsl.models.mlp_policy import RcslPolicy
    #     model = RcslPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
    #                     simple_input=args.simple_input)
        
    #     goal = train_dataset.get_max_return() * args.goal_mul

    #     import maze.algos.stitch_rcsl.training.trainer_mlp as trainer_mlp
    #     tconf = trainer_mlp.TrainerConfig(max_epochs=epochs, batch_size=args.batch, learning_rate=args.rate,
    #                 lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
    #                 desired_rtg=goal, ckpt_path = args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
    #                 ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb)
    #     output_policy_trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
    elif args.algo == 'rcsl-dt':
        from viskit.logging import logger as logger_dt, setup_logger as setup_logger_cql
        # logging_conf = WandBLogger.get_default_config(updates={"prefix":'stitch-mlp',
        #                                                        "project": 'stitch-rcsl',
        #                                                        "output_dir": './mlp_log'})
        # wandb_logger = WandBLogger(config=logging_conf, variant=vars(args))
        setup_logger_cql(
            variant=vars(args),
            exp_id=f"ctx{args.ctx}",
            seed=args.seed,
            base_log_dir="./rcsl-dt_log/",
            include_exp_prefix_sub_dir=False
        )
        setup_seed(args.seed)
        env.reset(seed=args.seed)

        offline_train_dataset = TrajCtxFloatLengthDataset(trajs, ctx = args.ctx, single_timestep = False, with_mask=True)
        # print(offline_train_dataset[275617])
        goal = offline_train_dataset.get_max_return() * args.goal_mul
        model = DecisionTransformer(
            state_dim=obs_dim,
            act_dim=action_dim,
            max_length=args.ctx,
            max_ep_len=args.horizon,
            action_tanh=False, # no tanh function
            hidden_size=args.embed_dim,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_inner=4*args.embed_dim,
            activation_function='relu',
            n_positions=1024,
            resid_pdrop=0.1,
            attn_pdrop=0.1)
        tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch, lr=args.rate,
                    lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                    desired_rtg=goal, ckpt_path= args.final_ckpt_path, env = env, tb_log = tb_dir_path, 
                    ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb, device=args.device, 
                    debug = args.debug, logger = logger_dt)
        output_policy_trainer = SequenceTrainer(
            config=tconf,
            model=model,
            offline_dataset=offline_train_dataset,
            is_gym = True)
    else:
        raise(Exception(f"Unimplemented model {args.algo}!"))

    print("Begin output policy training")
    output_policy_trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=123)
    parser.add_argument("--algo-name", type=str, default="dt_pickplace")
    parser.add_argument("--task", type=str, default="pickplace", help="pickplace, pickplace_easy") # Self-constructed environment

    # Overall configuration
    parser.add_argument('--maze_config_file', type=str, default='./config/maze2.json')
    parser.add_argument('--data_file', type=str, default='./dataset/maze2.dat')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    parser.add_argument('--env_type', type=str, default='pickplace', help='pointmaze or ?')
    parser.add_argument('--algo', type=str, default='rcsl-dt', help="rcsl-mlp, rcsl-dt or stitch-mlp, stitch-cql")
    # parser.add_argument('--horizon', type=int, default=200, help="Should be consistent with dataset")
    parser.add_argument('--num_workers', type=int, default=0, help="Dataloader workers")

    parser.add_argument('--data_dir', type=str, default='../../OfflineRL-Kit/dataset')
    parser.add_argument('--horizon', type=int, default=40, help="max path length for pickplace")

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

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--test_rollout", action='store_true')
    args = parser.parse_args()
    print(args)

    run(args=args)