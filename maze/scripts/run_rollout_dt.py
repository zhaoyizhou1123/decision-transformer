'''
MIT License

Copyright (c) 2021 Decision Transformer (Decision Transformer: Reinforcement Learning via Sequence Modeling) Authors (https://arxiv.org/abs/2106.01345)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

'''

import __init__

import gym
import numpy as np
import torch
import wandb
import time
import os

import argparse
import pickle
import random
import sys

from maze.algos.decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from maze.algos.decision_transformer.models.decision_transformer import DecisionTransformer
from maze.algos.decision_transformer.training.seq_trainer import SequenceTrainer

from torch.utils.tensorboard import SummaryWriter  
from maze.utils.dataset import TrajCtxDataset, TrajNextObsDataset, ObsActDataset
from maze.scripts.rollout import rollout_expand_trajs, test_rollout_combo, rollout_combo, test_rollout_diffusion, rollout_diffusion
from maze.algos.stitch_rcsl.training.trainer_dynamics import EnsembleDynamics 
from maze.algos.stitch_rcsl.training.trainer_base import TrainerConfig
from maze.algos.stitch_rcsl.training.train_mdp import BehaviorPolicyTrainer
from maze.algos.stitch_rcsl.models.mlp_policy import StochasticPolicy
from maze.algos.decision_transformer.models.decision_transformer import DecisionTransformer
from maze.algos.decision_transformer.training.seq_trainer import SequenceTrainer
from maze.utils.buffer import ReplayBuffer
from maze.utils.trajectory import Trajs2Dict
from maze.utils.scalar import StandardScaler
from maze.algos.stitch_rcsl.models.dynamics_normal import EnsembleDynamicsModel
from maze.utils.logger import Logger, make_log_dirs
from maze.utils.none_or_str import none_or_str
from maze.utils.setup_logger import setup_logger


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

def mb_rollout(args):
    '''
    Learn model and rollout
    '''
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
    # epochs = args.epochs

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
    # device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'
    device = args.device

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
        dynamics_trainer.train(real_buffer.sample_all(), logger, max_epochs_since_update=5, 
                               batch_size=args.dynamics_batch_size)   

    # Get behavior policy model 
    if args.behavior_type == 'mlp': # mlp model
        if args.mdp_ckpt_dir is not None and os.path.exists(os.path.join(args.mdp_ckpt_dir, "behavior.pth")):
            b_model_path = os.path.join(args.mdp_ckpt_dir, "behavior.pth")
            print(f"MDP model exists, load existing models from {b_model_path}")
            # d_model_path = os.path.join(args.mdp_ckpt_dir, "dynamics.pth")
            
            # i_model_pth = os.path.join(args.mdp_ckpt_dir, "init.pth")
            # dynamics_model = torch.load(d_model_path, map_location=device)
            behavior_model = torch.load(b_model_path, map_location=device)
            # init_model = torch.load(i_model_pth, map_location=device)
            # dynamics_model.train(False)
            behavior_model.train(False)
            # init_model.train(False)
        else: # Train behavior policy model
            if args.mdp_ckpt_dir is None:
                b_ckpt_path = os.path.join('./checkpoint',format_time)
            else:
                b_ckpt_path = args.mdp_ckpt_dir
            os.makedirs(b_ckpt_path,exist_ok=True)

            model_dataset = TrajNextObsDataset(trajs)
            print(f"Train behavior policy model")

            conf = TrainerConfig(max_epochs=args.behavior_epoch, batch_size=args.behavior_batch, learning_rate=args.rate,
                                num_workers=1, horizon=horizon, grad_norm_clip = 1.0, eval_repeat = 10,
                                ckpt_path = b_ckpt_path, tb_log = tb_dir_path, r_loss_weight = args.r_loss_weight,
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
                             load = args.load_diffusion,
                             save_ckpt_freq = 5,
                             device = args.device)
        
        state_action_dataset = ObsActDataset(trajs)
        # Configure logger
        diffusion_logger = setup_logger(conf)
        print(f"Setup logger")

        from maze.algos.stitch_rcsl.training.trainer_diffusion import DiffusionBC
        diffusion_policy = DiffusionBC(conf, state_action_dataset, diffusion_logger)

        if not args.load_diffusion:
            print(f"Train diffusion behavior policy")
            diffusion_policy.train() # save checkpoint periodically
            diffusion_policy.save_checkpoint(epoch=None) # Save final model
        else:
            print(f"Load diffusion behavior policy")
            diffusion_policy.load_checkpoint(final=True)

    else:
        raise Exception(f"Unimplemented behavior policy type {args.behavior_type}")

    if args.test_rollout:
        # Test rollout
        if args.behavior_type == 'mlp':
            test_rollout_combo(args, dynamics_trainer, real_buffer, behavior_model_raw=behavior_model, based_true_state=args.true_state, init_true_state=args.init_state)
        elif args.behavior_type == 'diffusion':
            print(f"Starting testing diffusion rollout")
            test_rollout_diffusion(args, dynamics_trainer, real_buffer, diffusion_policy, based_true_state=args.true_state, init_true_state=args.init_state)
            # test_rollout_combo(args, dynamics_trainer, real_buffer, behavior_model_raw=None, based_true_state=False, init_true_state=True)
    else:            
        # Get max dataset return
        traj_rets = [traj.returns[0] for traj in trajs]
        max_dataset_return = max(traj_rets)
        print(f"Maximum dataset return is {max_dataset_return}")

        if args.behavior_type == 'mlp':
            rollout_trajs = rollout_combo(args, dynamics_trainer, behavior_model, real_buffer,
                                          threshold=max_dataset_return)
        elif args.behavior_type == 'diffusion':
            rollout_trajs = rollout_diffusion(args, dynamics_trainer, diffusion_policy, real_buffer, 
                                              threshold=max_dataset_return)

    return rollout_trajs

def experiment(
        exp_prefix,
        variant,
        rollout_trajs = None
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

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
        state_dim = env.observation_space['observation'].shape[0]
        act_dim = env.action_space.shape[0]
        trajectories = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info

        assert len(trajectories[0].observations) == horizon, f"Horizon mismatch: {len(trajectories[0].observations)} and {horizon}"
        
        # Get dict type dataset, for dynamics training
        dynamics_dataset = Trajs2Dict(trajectories)
    else:
        raise Exception(f"Unimplemented env type {args.env_type}")

    # state_dim = env.observation_space.shape[0]
    # act_dim = env.action_space.shape[0]

    # load dataset
    # dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    # with open(dataset_path, 'rb') as f:
    #     trajectories = pickle.load(f)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        # if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
        #     path['rewards'][-1] = path['rewards'].sum()
        #     path['rewards'][:-1] = 0.
        states.append(np.array(path.observations))
        traj_lens.append(len(path.observations))
        returns.append(path.rewards.sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens) # Total timesteps of all trajs. num_traj * horizon

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj*num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    max_ep_len = args.epochs
    def get_batch(batch_size=256, max_len=K):
        '''
        max_len: ctx
        '''
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj.rewards.shape[0] - 1)

            # get sequences from dataset
            s.append(traj.observations[si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj.actions[si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj.rewards[si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1)) # real_len+1
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1] # Real length <= ctx
            # Pad in the front
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1)) # Padded entry with 0, other with 1

        # (batch, ctx, state/act_dim/1), (batch, ctx)
        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew/scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4*variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter+1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20) # ctx
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    parser.add_argument('--maze_config_file', type=str, default='./config/maze2.json')
    parser.add_argument('--data_file', type=str, default='./dataset/maze2.dat')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--log_to_wandb',action='store_true', help='Set up wandb')
    parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/, Folder to tensorboard logs" )
    parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    parser.add_argument('--algo', type=str, default='stitch-mlp', help="rcsl-mlp, rcsl-dt or stitch-mlp, stitch-dt")
    parser.add_argument('--horizon', type=int, default=200, help="Should be consistent with dataset")

    # Output policy
    parser.add_argument('--ctx', type=int, default=1)
    parser.add_argument('--pre_epochs', type=int, default=5, help='epochs to learn the output policy using offline data')
    parser.add_argument('--epochs', type=int, default=100, help='total epochs to learn the output policy')
    parser.add_argument('--model_type', type=str, default='reward_conditioned')
    # parser.add_argument('--num_steps', type=int, default=500000)
    # parser.add_argument('--num_buffers', type=int, default=50)
    # parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch', type=int, default=256, help='final output policy training batch')
    parser.add_argument('--final_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable. Used to store output policy model" )
    parser.add_argument('--arch', type=str, default='200-200-200-200', help="Hidden layer size of output mlp model" )
    parser.add_argument('--goal_mul', type=float, default=1, help="goal = max_dataset_return * goal_mul")
    
    
    
    # parser.add_argument('--mdp_batch', type=int, default=128)
    # parser.add_argument('--mdp_epochs', type=int, default=5, help='epochs to learn the mdp model')
    # parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    
    # parser.add_argument('--rollout_data_file', type=str, default=None, help='./dataset/maze_rollout.dat')
    # parser.add_argument('--log_level', type=str, default='WARNING')
    parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of all trainers" )
    parser.add_argument('--hash', action='store_true', help="Hash states if True")
    # parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
    # parser.add_argument('--env_path', type=str, default='./env/env_rev.txt', help='Path to env description file')
    parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension, default -1 for no embedding")
    parser.add_argument('--embed_dim', type=int, default=128, help="dt token embedding dimension")
    parser.add_argument('--n_layer', type=int, default=3, help="Transformer layer")
    parser.add_argument('--n_head', type=int, default=1, help="Transformer head")
    # parser.add_argument('--model', type=str, default='dt', help="mlp or dt")
    parser.add_argument('--n_support_init', type=int, default=5, help="Number of supporting initial states" )
    parser.add_argument('--d_arch', type=str, default='128', help="Hidden layer size of dynamics model" )
    parser.add_argument('--r_loss_weight', type=float, default=0.5, help="[0,1], weight of r_loss" )
    parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
    parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")
    parser.add_argument('--time_depend_s',action='store_true')
    parser.add_argument('--time_depend_a',action='store_true')
    parser.add_argument('--simple_input',action='store_false', help='Only use history rtg info if true')

    parser.add_argument("--seed", type=int, default=1)

    # Dynamics model
    parser.add_argument("--algo-name", type=str, default="combo")
    parser.add_argument("--dynamics-lr", type=float, default=1e-3)
    parser.add_argument("--dynamics-hidden-dims", type=int, nargs='*', default=[200, 200, 200, 200])
    parser.add_argument("--dynamics-weight-decay", type=float, nargs='*', default=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4])
    parser.add_argument("--n-ensemble", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    # parser.add_argument("--rollout-freq", type=int, default=1000)
    # parser.add_argument("--rollout-batch-size", type=int, default=50000)
    # parser.add_argument("--rollout-length", type=int, default=5)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.5)
    parser.add_argument("--load-dynamics-path", type=none_or_str, default=None)
    parser.add_argument("--dynamics-batch-size", type=int, default=256, help="batch size for dynamics training")

    # Behavior policy (diffusion)
    parser.add_argument('--behavior_type', type=str, default='diffusion', help='mlp or diffusion')
    parser.add_argument("--behavior_epoch", type=int, default=100)
    parser.add_argument("--num_diffusion_iters", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument('--behavior_batch', type=int, default=256)
    parser.add_argument('--load_diffusion', action="store_true", help="Load final diffusion policy model if true")
    parser.add_argument('--diffusion_seed', type=str, default='0', help="Distinguish runs for diffusion policy")

    # Behavior policy (mlp)
    parser.add_argument('--mdp_ckpt_dir', type=str, default='./checkpoint/mlp_b_policy', help="dir path, used to load/store mlp behavior policy model" )
    parser.add_argument('--b_arch', type=str, default='200-200-200-200', help="Hidden layer size of behavior model" )
    parser.add_argument('--n_support', type=int, default=10, help="Number of supporting action of policy" )

    # Rollout
    parser.add_argument('--true_state', action="store_true", help="Rollout based_true_state")
    parser.add_argument('--init_state', action="store_true", help="Rollout init_true_state")
    parser.add_argument('--rollout_ckpt_path', type=none_or_str, default=None, help="./checkpoint/maze2_smd_stable, file path, used to load/store rollout trajs" )
    parser.add_argument('--rollout_epochs', type=int, default=1000, help="Max number of epochs to rollout the policy")
    parser.add_argument('--num_need_traj', type=int, default=100, help="Needed valid trajs in rollout")

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--test_rollout", action='store_true')
    args = parser.parse_args()
    print(args)

    experiment('gym-experiment', variant=vars(args))
