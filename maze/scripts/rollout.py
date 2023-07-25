# Roll out policy based on dataset modeling
import __init__

import argparse
from maze.utils.sample import sample_from_supports
from maze.utils.trajectory import Trajectory
import pickle
import time
import os
import wandb
import torch
from copy import deepcopy
from maze.policies.maze_expert import WaypointController
from maze.utils.buffer import ReplayBuffer
from maze.algos.stitch_rcsl.training.trainer_base import BaseDynamics
import numpy as np

def test_rollout(env, dynamics_model, behavior_model, init_model, device, args, based_true_state = False, init_true_state = False):
    '''
    Still testing, use the true env to rollout. We try to see whether we can get optimal trajectory
    using the learned policy model.
    based_true_state: If True, action sampling is based on true_state; Otherwise pred_state
    init_true_state: If True, sampler knows the true init_state
    '''
    for epoch in range(args.rollout_epochs):
        true_state, _ = env.reset()
        if hasattr(env, 'get_true_observation'): # For pointmaze
            true_state = env.get_true_observation(true_state)
        true_state = torch.from_numpy(true_state)
        true_state = true_state.type(torch.float32).to(device) # (state_dim)

        if init_true_state:
            pred_state = deepcopy(true_state)
        else:
            pred_states, pred_state_probs = init_model(true_state) # use cuda:0
            pred_state = sample_from_supports(pred_states, pred_state_probs)

        # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
        print(f"-----------\nEpoch {epoch}")

        ret = 0 # total return 
        pred_ret = 0
        for h in range(args.horizon):
            timestep = torch.tensor(h).to(device) # scalar
            # Get action
            # pred_actions = self.model(states, actions, rtgs, timesteps) #(1, action_dim)
            if based_true_state:
                support_actions, support_probs = behavior_model(true_state.unsqueeze(dim=0)) # (1, n_support, action_dim), (1,n_support)
            else:
                support_actions, support_probs = behavior_model(pred_state.unsqueeze(dim=0)) # (1, n_support, action_dim), (1,n_support)
            # sample_idx = torch.multinomial(support_probs, num_samples=1).squeeze() # scalar
            # action = support_actions[0,sample_idx,:] # (action_dim)
            action = sample_from_supports(support_actions.squeeze(0), support_probs.squeeze(0))
            # print(action)
            pred_reward, pred_next_state = dynamics_model(pred_state, action, timestep=None)
            # Observe next states, rewards,
            next_state, reward, terminated, _, _ = env.step(action.detach().cpu().numpy()) # array (state_dim), scalar
            if hasattr(env, 'get_true_observation'): # For pointmaze
                next_state = env.get_true_observation(next_state)
            next_state = torch.from_numpy(next_state) # (state_dim)
            print(f"Step {h}")
            print(f"True reward: {reward}, predicted {pred_reward}")
            print(f"True next_state: {next_state}, predicted {pred_next_state}\n")
            # Calculate return
            ret += reward
            pred_ret += pred_reward
            
            # Update states, actions, rtgs, timesteps
            true_state = next_state.to(device).type(torch.float32) # (state_dim)
            pred_state = pred_next_state.type(torch.float32)

            # Update timesteps

            if terminated: # Already reached goal, the rest steps get reward 1, break
                ret += args.horizon - 1 - h
                pred_ret += args.horizon - 1 -h
                break
        print(f"Total return: {ret}, predicted total return {pred_ret}")
        # Add the ret to list
        # rets.append(ret)

def test_rollout_inv(args, based_true_state = False, init_true_state = False):
    '''
    Still testing, use the true env to rollout. We try to see whether we can get optimal trajectory
    using the learned policy model.
    based_true_state: To be implemented
    '''
    d_path = os.path.join(args.ckpt_dir, "inv_dynamics.pth")
    p_path = os.path.join(args.ckpt_dir, "next_state.pth")
    r_path = os.path.join(args.ckpt_dir, "reward.pth")
    i_path = os.path.join(args.ckpt_dir, "init.pth")

    device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'

    inv_dynamics_model = torch.load(d_path, map_location=device)
    next_state_model = torch.load(p_path, map_location=device)
    reward_model = torch.load(r_path, map_location=device)
    init_model = torch.load(i_path, map_location=device)


    inv_dynamics_model.train(False)
    next_state_model.train(False)
    reward_model.train(False)
    init_model.train(False)


    if args.env_type == 'pointmaze': # Create env and dataset
        from maze.scripts.create_maze_dataset import create_env
        env = create_env(args)

        # Add a get_true_observation method for Env
        def get_true_observation(obs):
            '''
            obs, obs received from pointmaze Env
            '''
            return obs['observation']
    
        setattr(env, 'get_true_observation', get_true_observation)

        # horizon = args.horizon
        # obs_dim = env.observation_space['observation'].shape[0]
        # action_dim = env.action_space.shape[0]
        # trajs = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info
        # assert len(trajs[0].observations) == horizon, f"Horizon mismatch: {len(trajs[0].observations)} and {horizon}"
    else:
        raise Exception(f"Unimplemented env type {args.env_type}")
    
    # if args.policy_type == 'model':
    #     b_model_path = os.path.join(args.ckpt_dir, "behavior.pth")
    #     behavior_model = torch.load(b_model_path, map_location=device)
    #     behavior_model.train(False)
    # elif args.policy_type == 'expert':
    #     init_obs, _ = env.reset() # Get desired_goal
    #     desired_goal = init_obs['desired_goal']
    #     def expert_policy(state):
    #         '''
    #         state: Tensor (1,state_dim)
    #         Output: support_action Tensor (1,1,action_dim), support probs Tensor([[1]])
    #         '''
    #         device = state.device
    #         state = state.squeeze(0).detach().cpu().numpy() # (4,)
    #         # create obs for expert_sampler
    #         achieved_goal = state[0:2] # pos
    #         obs = {'observation': state, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}
    #         print(f'obs = {obs}')
    #         expert_sampler = WaypointController(maze = deepcopy(env.maze))
    #         action = expert_sampler.compute_action(obs) # np.array(action_dim)
    #         action = torch.from_numpy(action).to(device).unsqueeze(0).unsqueeze(0) # (1,1,action_dim)

    #         return action.type(torch.float32), torch.tensor([[1.]]).to(device)
    #     behavior_model = expert_policy
    # else:
    #     raise Exception(f"Unknown policy_type {args.policy_type}")

    # Rollout begins
    for epoch in range(args.rollout_epochs):
        true_state, _ = env.reset()
        if hasattr(env, 'get_true_observation'): # For pointmaze
            true_state = env.get_true_observation(true_state)
        true_state = torch.from_numpy(true_state)
        true_state = true_state.type(torch.float32).to(device) # (state_dim)

        if init_true_state:
            pred_state = deepcopy(true_state)
        else:
            pred_states, pred_state_probs = init_model(true_state) # use cuda:0
            pred_state = sample_from_supports(pred_states, pred_state_probs)

        # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
        print(f"-----------\nEpoch {epoch}")

        ret = 0 # total return 
        pred_ret = 0
        for h in range(args.horizon):
            timestep = torch.tensor(h).to(device) # scalar
            # Get action
            # pred_actions = self.model(states, actions, rtgs, timesteps) #(1, action_dim)
            if based_true_state: # Not clear how to implement
                support_next_states, support_probs = next_state_model(true_state.unsqueeze(dim=0)) # (1, n_support, state_dim), (1,n_support)
            else:
                support_next_states, support_probs = next_state_model(pred_state.unsqueeze(dim=0)) # (1, n_support, action_dim), (1,n_support)
            # sample_idx = torch.multinomial(support_probs, num_samples=1).squeeze() # scalar
            # action = support_actions[0,sample_idx,:] # (action_dim)
            pred_next_state = sample_from_supports(support_next_states.squeeze(0), support_probs.squeeze(0))
            # print(action)
            if based_true_state:
                pred_action = inv_dynamics_model(true_state, pred_next_state)
                pred_reward = reward_model(true_state, pred_action)
            else:
                pred_action = inv_dynamics_model(pred_state, pred_next_state)
                pred_reward = reward_model(pred_state, pred_action)
            # Observe next states, rewards,
            next_state, reward, terminated, _, _ = env.step(pred_action.detach().cpu().numpy()) # array (state_dim), scalar
            if hasattr(env, 'get_true_observation'): # For pointmaze
                next_state = env.get_true_observation(next_state)
            next_state = torch.from_numpy(next_state) # (state_dim)
            print(f"Step {h}, pred_action {pred_action.detach().cpu().numpy()}")
            print(f"True reward: {reward}, predicted {pred_reward}")
            print(f"True next_state: {next_state}, predicted {pred_next_state}\n")
            # Calculate return
            ret += reward
            pred_ret += pred_reward
            
            # Update states, actions, rtgs, timesteps
            true_state = next_state.to(device).type(torch.float32) # (state_dim)
            pred_state = pred_next_state.type(torch.float32)

            # Update timesteps

            if terminated: # Already reached goal, the rest steps get reward 1, break
                ret += args.horizon - 1 - h
                pred_ret += args.horizon - 1 -h
                break
        print(f"Total return: {ret}, predicted total return {pred_ret}")
        # Add the ret to list
        # rets.append(ret)

def test_rollout_combo(args, dynamics: BaseDynamics, dynamics_dataset: ReplayBuffer, behavior_model_raw = None, based_true_state = False, init_true_state = False):
    '''
    Combo style dynamics rollout
    '''
    device = args.device
    # args.horizon=400
    # Set env
    if args.env_type == 'pointmaze': # Create env and dataset
        from maze.scripts.create_maze_dataset import create_env
        env = create_env(args)

        # Add a get_true_observation method for Env
        def get_true_observation(obs):
            '''
            obs, obs received from pointmaze Env
            '''
            return obs['observation']
    
        setattr(env, 'get_true_observation', get_true_observation)

        # horizon = args.horizon
        # obs_dim = env.observation_space['observation'].shape[0]
        # action_dim = env.action_space.shape[0]
        # trajs = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info
        # assert len(trajs[0].observations) == horizon, f"Horizon mismatch: {len(trajs[0].observations)} and {horizon}"
    else:
        raise Exception(f"Unimplemented env type {args.env_type}")
    
    if behavior_model_raw is None:
        # Get expert behavior model
        init_obs_temp, _ = env.reset() # Get desired_goal
        desired_goal = init_obs_temp['desired_goal']
        # desired_goal = np.array([4,-1.3])
        print(f"Desired goal: {desired_goal}")
        def expert_policy(state):
            '''
            state: np.array (state_dim)
            Output: support_action Tensor (1,1,action_dim), support probs Tensor([[1]])
            '''
            # device = state.device
            # state = state.squeeze(0).detach().cpu().numpy() # (4,)
            # create obs for expert_sampler
            achieved_goal = state[0:2] # pos
            obs = {'observation': state, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}
            # print(f'obs = {obs}')
            expert_sampler = WaypointController(maze = deepcopy(env.maze))
            action = expert_sampler.compute_action(obs) # np.array(action_dim)
            action = torch.from_numpy(action).to(device).unsqueeze(0).unsqueeze(0) # (1,1,action_dim)

            return action.type(torch.float32), torch.tensor([[1.]]).to(device)
        behavior_model = expert_policy
    else:
        def model_policy(state):
            # state: np.array (state_dim)
            state = torch.from_numpy(state)
            return behavior_model_raw(state.unsqueeze(0).to(device)) # (1, n_support, action_dim), (1,n_support)
        behavior_model = model_policy

    # Run epochs
    for epoch in range(args.rollout_epochs):
        true_state, _ = env.reset()
        if hasattr(env, 'get_true_observation'): # For pointmaze
            true_state = env.get_true_observation(true_state)
        # true_state = torch.from_numpy(true_state)
        # true_state = true_state.type(torch.float32).to(device) # (state_dim)

        if init_true_state:
            pred_state = deepcopy(true_state) # np.array
        else:
            pred_state = dynamics_dataset.sample_init_obs().detach().cpu().numpy() # np.array
            # print(pred_state)

        # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
        print(f"-----------\nEpoch {epoch}")

        ret = 0 # total return 
        pred_ret = 0
        for h in range(args.horizon):
            timestep = torch.tensor(h).to(device) # scalar
            # Get action
            # pred_actions = self.model(states, actions, rtgs, timesteps) #(1, action_dim)
            if based_true_state:
                support_actions, support_probs = behavior_model(true_state) # (1, n_support, action_dim), (1,n_support)
            else:
                support_actions, support_probs = behavior_model(pred_state) # (1, n_support, action_dim), (1,n_support)
            # sample_idx = torch.multinomial(support_probs, num_samples=1).squeeze() # scalar
            # action = support_actions[0,sample_idx,:] # (action_dim)
            action = sample_from_supports(support_actions.squeeze(0), support_probs.squeeze(0)).detach().cpu().numpy()
            # print(action)
            pred_next_state, pred_reward, _, _ = dynamics.step(pred_state, action) # (1,state_dim), (1,1)
            pred_next_state = pred_next_state.squeeze(0) # (state_dim)
            pred_reward = pred_reward.squeeze() # scalar
            # print(pred_next_state.shape, pred_reward.shape)
            # Observe next states, rewards,
            next_state, reward, terminated, _, _ = env.step(action) # array (state_dim), scalar
            if hasattr(env, 'get_true_observation'): # For pointmaze
                next_state = env.get_true_observation(next_state)
            # next_state = torch.from_numpy(next_state) # (state_dim)
            if args.debug:
                print(f"Step {h}, action {action}")
                print(f"True reward: {reward}, predicted {pred_reward}")
                print(f"True next_state: {next_state}, predicted {pred_next_state}\n")
            # Calculate return
            ret += reward
            pred_ret += pred_reward
            
            # Update states, actions, rtgs, timesteps
            true_state = next_state # (state_dim)
            pred_state = pred_next_state

            # Update timesteps

            # if terminated: # Already reached goal, the rest steps get reward 1, break
            #     ret += args.horizon - 1 - h
            #     pred_ret += args.horizon - 1 -h
            #     break
        print(f"Total return: {ret}, predicted total return {pred_ret}")

def rollout_expand_trajs(dynamics_model, behavior_model, init_model, device, args):
    '''
    Rollout and create new trajs.
    Return: list(Traj), new created trajs. But don't have terminated, truncated, set to None instead.
    infos set to list, element fixed to 'rollout', to tell dataset it is the rollout traj. \n
    Data type same as original Traj data.

    '''
    trajs = []
    for epoch in range(args.rollout_epochs):
        support_init_states, support_init_probs = init_model() # (n_support, state_dim), (n_support)
        state = sample_from_supports(support_init_states, support_init_probs)
        state = state.type(torch.float32).to(device) # (state_dim)
        

        # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
        print(f"-----------\nEpoch {epoch}")

        # pred_ret = 0
        achieved_ret = 0

        observations_ = []
        actions_ = []
        rewards_ = []
        achieved_rets_ = [] # The total reward that has achieved, used to compute rtg

        for h in range(args.horizon):
            timestep = torch.tensor(h).to(device) # scalar

            # Record, obs, achieved_ret, timestep
            observations_.append(deepcopy(state.detach().cpu().numpy()))
            achieved_rets_.append(deepcopy(achieved_ret))

            # Get action
            # pred_actions = self.model(states, actions, rtgs, timesteps) #(1, action_dim)
            support_actions, support_probs = behavior_model(state.unsqueeze(dim=0)) # (1, n_support, action_dim), (1,n_support)
            action = sample_from_supports(support_actions.squeeze(0), support_probs.squeeze(0))
            pred_reward, pred_next_state = dynamics_model(state, action, timestep)

            actions_.append(deepcopy(action.detach().cpu().numpy()))
            rewards_.append(deepcopy(pred_reward.item()))

            # Calculate return
            achieved_ret += pred_reward.item()

            
            # Update states, actions, rtgs, timesteps
            state = pred_next_state.to(device).type(torch.float32) # (state_dim)
        pred_ret = achieved_ret
        print(f"Predicted total return {pred_ret}")

        returns_ = [pred_ret - achieved for achieved in achieved_rets_]

        # Add new traj
        trajs.append(Trajectory(observations = observations_, 
                                actions = actions_, 
                                rewards = rewards_, 
                                returns = returns_, 
                                timesteps = list(range(args.horizon)), 
                                terminated = None, 
                                truncated = None, 
                                infos = ['rollout' for _ in range(args.horizon)]))
        
    return trajs

def rollout_combo(args, dynamics: BaseDynamics, behavior_model, dynamics_dataset: ReplayBuffer):
    '''
    Combo style dynamics rollout
    '''
    device = args.device
    # args.horizon=400
    # Set env
    
    # # Get expert behavior model
    # init_obs_temp, _ = env.reset() # Get desired_goal
    # desired_goal = init_obs_temp['desired_goal']
    # # desired_goal = np.array([4,-1.3])
    # print(f"Desired goal: {desired_goal}")
    # def expert_policy(state):
    #     '''
    #     state: np.array (state_dim)
    #     Output: support_action Tensor (1,1,action_dim), support probs Tensor([[1]])
    #     '''
    #     # device = state.device
    #     # state = state.squeeze(0).detach().cpu().numpy() # (4,)
    #     # create obs for expert_sampler
    #     achieved_goal = state[0:2] # pos
    #     obs = {'observation': state, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}
    #     # print(f'obs = {obs}')
    #     expert_sampler = WaypointController(maze = deepcopy(env.maze))
    #     action = expert_sampler.compute_action(obs) # np.array(action_dim)
    #     action = torch.from_numpy(action).to(device).unsqueeze(0).unsqueeze(0) # (1,1,action_dim)

    #     return action.type(torch.float32), torch.tensor([[1.]]).to(device)
    # behavior_model = expert_policy

    # Run epochs
    trajs = []
    for epoch in range(args.rollout_epochs):
        # true_state, _ = env.reset()
        # if hasattr(env, 'get_true_observation'): # For pointmaze
        #     true_state = env.get_true_observation(true_state)
        # true_state = torch.from_numpy(true_state)
        # true_state = true_state.type(torch.float32).to(device) # (state_dim)

        # if init_true_state:
        #     pred_state = deepcopy(true_state) # np.array
        # else:
        pred_state = dynamics_dataset.sample_init_obs() # tensor
            # print(pred_state)

        # print(f"Eval forward: states {states.shape}, actions {actions.shape}")
        print(f"-----------\nEpoch {epoch}")

        pred_ret = 0

        observations_ = []
        actions_ = []
        rewards_ = []
        achieved_rets_ = [] # The total reward that has achieved, used to compute rtg
        for h in range(args.horizon):
            timestep = torch.tensor(h).to(device) # scalar
            observations_.append(deepcopy(pred_state.detach().cpu().numpy()))
            achieved_rets_.append(deepcopy(pred_ret))
            # Get action
            # pred_actions = self.model(states, actions, rtgs, timesteps) #(1, action_dim)
            # if based_true_state:
            #     support_actions, support_probs = behavior_model(true_state) # (1, n_support, action_dim), (1,n_support)
            # else:
            support_actions, support_probs = behavior_model(pred_state.unsqueeze(0).to(device)) # (1, n_support, action_dim), (1,n_support)
            # sample_idx = torch.multinomial(support_probs, num_samples=1).squeeze() # scalar
            # action = support_actions[0,sample_idx,:] # (action_dim)
            action = sample_from_supports(support_actions.squeeze(0), support_probs.squeeze(0)).detach().cpu().numpy()
            # print(action)
            pred_next_state, pred_reward, _, _ = dynamics.step(pred_state.detach().cpu().numpy(), action) # (1,state_dim), (1,1)
            pred_next_state = pred_next_state.squeeze(0) # (state_dim)
            pred_reward = pred_reward.squeeze() # scalar
            # print(pred_next_state.shape, pred_reward.shape)
            # Observe next states, rewards,
            # next_state, reward, terminated, _, _ = env.step(action) # array (state_dim), scalar
            # if hasattr(env, 'get_true_observation'): # For pointmaze
            #     next_state = env.get_true_observation(next_state)
            # next_state = torch.from_numpy(next_state) # (state_dim)
            actions_.append(deepcopy(action))
            rewards_.append(deepcopy(pred_reward))
            if args.debug:
                print(f"Step {h}, action {action}")
                print(f"Predicted reward {pred_reward}")
                print(f"Predicted state {pred_next_state}\n")
            # Calculate return
            # ret += reward
            pred_ret += pred_reward
            
            # Update states, actions, rtgs, timesteps
            # true_state = next_state # (state_dim)
            pred_state = torch.from_numpy(pred_next_state)

            # Update timesteps

            # if terminated: # Already reached goal, the rest steps get reward 1, break
            #     ret += args.horizon - 1 - h
            #     pred_ret += args.horizon - 1 -h
            #     break
        returns_ = [pred_ret - achieved for achieved in achieved_rets_]
        print(f"Epoch {epoch}, predicted total return {pred_ret}")

        # Add new traj
        trajs.append(Trajectory(observations = observations_, 
                                actions = actions_, 
                                rewards = rewards_, 
                                returns = returns_, 
                                timesteps = list(range(args.horizon)), 
                                terminated = None, 
                                truncated = None, 
                                infos = ['rollout' for _ in range(args.horizon)]))
        
    return trajs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--rollout_epochs', type=int, default=100, help="Number of epochs to rollout the policy")
    # parser.add_argument('--num_steps', type=int, default=500000)
    # parser.add_argument('--num_buffers', type=int, default=50)
    # parser.add_argument('--game', type=str, default='Breakout')
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--env_type', type=str, default='pointmaze', help='pointmaze or ?')
    # parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--rollout_data_file', type=str, default=None, help='./dataset/maze_rollout.dat')
    parser.add_argument('--horizon', type=int, default=400, help="Should be consistent with dataset!")
    parser.add_argument('--ckpt_dir', type=str, default="./checkpoint/support2policy_07121318", help="./checkpoint/[dir_name], path to specific checkpoint dir" )
    parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of Trainer" )
    parser.add_argument('--tb_path', type=str, default=None, help="./logs/stitch/" )
    parser.add_argument('--n_embd', type=int, default=-1, help="token embedding dimension, default -1 for no embedding")
    parser.add_argument('--algo', type=str, default='rcsl-mlp', help="rcsl-mlp or ?")
    parser.add_argument('--b_arch', type=str, default='128', help="Hidden layer size of behavior model" )
    parser.add_argument('--d_arch', type=str, default='128', help="Hidden layer size of dynamics model" )
    parser.add_argument('--repeat', type=int, default=1, help="Repeat tokens in Q-network")
    parser.add_argument('--sample', action='store_false', help="Sample action by probs, or choose the largest prob")
    parser.add_argument('--time_depend_s',action='store_true')
    parser.add_argument('--time_depend_a',action='store_true')
    parser.add_argument('--simple_input',action='store_false', help='Only use history rtg info if true')
    parser.add_argument('--log_to_wandb',action='store_false', help='Set up wandb')
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--maze_config_file', type=str, default='./config/maze1.json')
    parser.add_argument('--policy_type', type=str, default='model', help='expert or model. Former uses expert sampling policy.')
    parser.add_argument('--train_type', type=str, default='default', help="default, inverse")
    args = parser.parse_args()
    print(args)

    if args.train_type == 'default':
        d_model_path = os.path.join(args.ckpt_dir, "dynamics.pth")
        i_model_pth = os.path.join(args.ckpt_dir, "init.pth")

        device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else 'cpu'

        dynamics_model = torch.load(d_model_path, map_location=device)
        init_model = torch.load(i_model_pth, map_location=device)


        dynamics_model.train(False)
        init_model.train(False)


        if args.env_type == 'pointmaze': # Create env and dataset
            from maze.scripts.create_maze_dataset import create_env
            env = create_env(args)

            # Add a get_true_observation method for Env
            def get_true_observation(obs):
                '''
                obs, obs received from pointmaze Env
                '''
                return obs['observation']
        
            setattr(env, 'get_true_observation', get_true_observation)

            # horizon = args.horizon
            # obs_dim = env.observation_space['observation'].shape[0]
            # action_dim = env.action_space.shape[0]
            # trajs = point_maze_offline.dataset[0] # the first element is trajs, the rest are debugging info
            # assert len(trajs[0].observations) == horizon, f"Horizon mismatch: {len(trajs[0].observations)} and {horizon}"
        else:
            raise Exception(f"Unimplemented env type {args.env_type}")
        
        if args.policy_type == 'model':
            b_model_path = os.path.join(args.ckpt_dir, "behavior.pth")
            behavior_model = torch.load(b_model_path, map_location=device)
            behavior_model.train(False)
        elif args.policy_type == 'expert':
            init_obs, _ = env.reset() # Get desired_goal
            desired_goal = init_obs['desired_goal']
            def expert_policy(state):
                '''
                state: Tensor (1,state_dim)
                Output: support_action Tensor (1,1,action_dim), support probs Tensor([[1]])
                '''
                device = state.device
                state = state.squeeze(0).detach().cpu().numpy() # (4,)
                # create obs for expert_sampler
                achieved_goal = state[0:2] # pos
                obs = {'observation': state, 'achieved_goal': achieved_goal, 'desired_goal': desired_goal}
                # print(f'obs = {obs}')
                expert_sampler = WaypointController(maze = deepcopy(env.maze))
                action = expert_sampler.compute_action(obs) # np.array(action_dim)
                action = torch.from_numpy(action).to(device).unsqueeze(0).unsqueeze(0) # (1,1,action_dim)

                return action.type(torch.float32), torch.tensor([[1.]]).to(device)
            behavior_model = expert_policy
        else:
            raise Exception(f"Unknown policy_type {args.policy_type}")
        cur_time = time.localtime(time.time())
        format_time = f"{cur_time.tm_mon:02d}{cur_time.tm_mday:02d}{cur_time.tm_hour:02d}{cur_time.tm_min:02d}"

        test_rollout(env, dynamics_model, behavior_model, init_model, device, args, based_true_state=True, init_true_state=True)
    elif args.train_type == 'inverse':
        test_rollout_inv(args, based_true_state=True, init_true_state=True)