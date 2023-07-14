import __init__

import argparse
import os
import pickle
import wandb
import time

from torch.utils.tensorboard import SummaryWriter  
from maze.utils.dataset import TrajCtxDataset
from maze.scripts.rollout import rollout_expand_trajs
from maze.scripts.learn_mdp_main import learn_mdp
import torch

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
        obs_dim = env.observation_space['observation'].shape[0]
        action_dim = env.action_space.shape[0]
        trajs = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info
        assert len(trajs[0].observations) == horizon, f"Horizon mismatch: {len(trajs[0].observations)} and {horizon}"
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
    if args.tb_path is not None:
        # Create tb log dir
        data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"
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
    if args.rollout_ckpt_path is not None and os.path.exists(args.rollout_ckpt_path):
        print("Rollout trajs exist, load rollout trajs")
        with open(args.rollout_ckpt_path, "rb") as f:
            rollout_trajs = pickle.load(args.rollout_ckpt_path)
    # Load fitted model if possible
    elif args.mdp_ckpt_dir is not None and os.path.exists(args.mdp_ckpt_dir):
        print("MDP model exists, load existing models")
        d_model_path = os.path.join(args.mdp_ckpt_dir, "dynamics.pth")
        b_model_path = os.path.join(args.mdp_ckpt_dir, "behavior.pth")
        i_model_pth = os.path.join(args.mdp_ckpt_dir, "init.pth")
        dynamics_model = torch.load(d_model_path, map_location=device)
        behavior_model = torch.load(b_model_path, map_location=device)
        init_model = torch.load(i_model_pth, map_location=device)
        dynamics_model.train(False)
        behavior_model.train(False)
        init_model.train(False)

        print("Rollout")
        rollout_trajs = rollout_expand_trajs(dynamics_model, behavior_model, init_state_model,
                                            device,args)
    # Fit model, rollout
    else:
        # Fit the model
        print(f"Fit the MDP model.")
        if tb_dir_path is not None:
            learn_mdp_tb_dir_path = os.path.join(tb_dir_path, 'model')
        else:
            learn_mdp_tb_dir_path = None
        dynamics_model, behavior_model, init_state_model = learn_mdp(trajs, 
                                                                    obs_dim, 
                                                                    action_dim, 
                                                                    learn_mdp_tb_dir_path, 
                                                                    args.mdp_ckpt_dir, 
                                                                    args)
        
        # Rollout the learned behavior policy using the learned dynamics model
        print("Rollout")
        rollout_trajs = rollout_expand_trajs(dynamics_model, behavior_model, init_state_model,
                                            device,args)
    
    # Learn the output policy using expanded dataset
    print("Fit output policy")
    train_dataset = TrajCtxDataset(trajs + rollout_trajs, ctx = args.ctx, single_timestep = False)
    if args.debug:
        print(train_dataset.len())
    # print("Finish generation")

    # Set models
    # if args.algo == 'dt':
    #     # Set GPT parameters
    #     n_layer = args.n_layer
    #     n_head = args.n_head
    #     n_embd = args.n_embd
    #     print(f"GPTConfig: n_layer={n_layer}, n_head={n_head}, n_embd={n_embd}")

    #     # print("Begin GPT configuartion.")
    #     mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
    #                     n_layer=n_layer, n_head=n_head, n_embd=n_embd, model_type=args.algo_type, max_timestep=horizon)
    #     # print("End GPT config, begin model generation")
    #     model = GPT(mconf)
    #     # print("End model generation")
    if args.algo == 'rcsl-mlp':
        # num_action = env.get_num_action()
        # print(f"num_action={num_action}")
        # model = MlpPolicy(1, num_action, args.ctx, horizon, args.repeat, args.arch, args.n_embd)
        from maze.algos.rcsl.models.mlp_policy import MlpPolicy

        model = MlpPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
                        simple_input=args.simple_input)
    elif args.algo == 'stitch':
        from maze.algos.stitch_rcsl.models.mlp_policy import RcslPolicy
        model = RcslPolicy(obs_dim, action_dim, args.ctx, horizon, args.repeat, args.arch, args.n_embd,
                        simple_input=args.simple_input)
    else:
        raise(Exception(f"Unimplemented model {args.algo}!"))

    # print("Begin Trainer configuartion")
    # if args.algo == 'dt':
    #     tconf = trainer_toy.TrainerConfig(max_epochs=epochs, batch_size=args.batch, learning_rate=args.rate,
    #                 lr_decay=True, warmup_tokens=512*20, final_tokens=2*train_dataset.len()*args.ctx*3,
    #                 num_workers=1, model_type=args.algo_type, max_timestep=horizon, horizon=horizon, 
    #                 desired_rtg=args.goal, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
    #                 ctx = args.ctx)
    #     print("End trainer configuration, begin trainer generation")
    #     trainer = trainer_toy.Trainer(model, train_dataset, None, tconf)
    if args.algo == 'rcsl-mlp':
        goal = train_dataset.get_max_return() * args.goal_mul

        import maze.algos.rcsl.trainer_mlp as trainer_mlp
        tconf = trainer_mlp.TrainerConfig(max_epochs=epochs, batch_size=args.batch, learning_rate=args.rate,
                    lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                    desired_rtg=goal, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
                    ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb)
        trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
    elif args.algo == 'stitch':
        goal = train_dataset.get_max_return() * args.goal_mul

        import maze.algos.stitch_rcsl.training.trainer_mlp as trainer_mlp
        tconf = trainer_mlp.TrainerConfig(max_epochs=epochs, batch_size=args.batch, learning_rate=args.rate,
                    lr_decay=True, num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                    desired_rtg=goal, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
                    ctx = args.ctx, sample = args.sample, log_to_wandb = args.log_to_wandb)
        trainer = trainer_mlp.Trainer(model, train_dataset, tconf)
    else:
        raise Exception(f"Unimplemented model type {args.algo}")

    # print("End trainer generation. Begin training.")
    trainer.train()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--ctx', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10, help='epochs to learn the output policy')
    parser.add_argument('--mdp_epochs', type=int, default=5, help='epochs to learn the mdp model')
    parser.add_argument('--rollout_epochs', type=int, default=10, help="Number of epochs to rollout the policy")
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    # parser.add_argument('--num_steps', type=int, default=500000)
    # parser.add_argument('--num_buffers', type=int, default=50)
    # parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--mdp_batch', type=int, default=128)
    
    # parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--data_file', type=str, default='./dataset/maze.dat')
    parser.add_argument('--rollout_data_file', type=str, default=None, help='./dataset/maze_rollout.dat')
    parser.add_argument('--log_level', type=str, default='WARNING')
    parser.add_argument('--goal_mul', type=float, default=1, help="goal = max_dataset_return * goal_mul")
    parser.add_argument('--horizon', type=int, default=250, help="Should be consistent with dataset")
    parser.add_argument('--ckpt_prefix', type=str, default=None, help="Used to store output policy model" )
    parser.add_argument('--mdp_ckpt_dir', type=str, default=None, help="dir path, used to load/store mdp model" )
    parser.add_argument('--rollout_ckpt_path', type=str, default=None, help="file path, used to load/store rollout trajs" )
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
    parser.add_argument('--log_to_wandb',action='store_false', help='Set up wandb')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    args = parser.parse_args()
    print(args)

    run(args=args)