# Run expert policy

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
from maze.utils.maze_utils import set_map_cell
import numpy as np
import json
import gymnasium as gym
from maze.samplers.maze_sampler import MazeSampler

def _collect_single_traj(self, start, goal, repeat, random_end: bool):
    '''
    Collect one trajectory.
    start: np.array (2,), type=int. Initial (row,col)
    goal: np.array (2,), type=int. Goal (row,col)
    repeat: int. times to repeat
    random_end: If True, do random walk when goal reached
    Return: Trajectory
    '''

    # Configure behavior maze map, the map for controller to take action
    behavior_map = set_map_cell(self.MAZE_MAP, start, 'r')
    behavior_map = set_map_cell(behavior_map, goal, 'g')

    # Set up behavior environment
    if self.render:
        render_mode = 'human'
    else:
        render_mode = None
    # render_mode = None

    # print(f"Behavior_env render mode {render_mode}")
    behavior_env = gym.make('PointMaze_UMazeDense-v3', 
                            maze_map= behavior_map, 
                            continuing_task = False,
                            max_episode_steps=self.horizon,
                            render_mode = render_mode)

    # Set up controller
    controller = WaypointController(maze = deepcopy(behavior_env.maze))

    # Configure data maze and env, the env to record data
    data_map = set_map_cell(self.MAZE_MAP, start, 'r')
    data_map = set_map_cell(data_map, self.target_goal, 'g')

    data_env = gym.make('PointMaze_UMazeDense-v3', 
                            maze_map= data_map, 
                            continuing_task = False,
                            max_episode_steps=self.horizon)
    
    # if self.debug:
    #     print(f"behavior_env==data_env: {behavior_env==data_env}")
    
    # Initialize data to record
    trajs = []
    for _ in range(repeat):
        observations_ = []
        actions_ = []
        rewards_ = []
        achieved_rets_ = [] # The total reward that has achieved, used to compute rtg
        timesteps_ = []
        terminateds_ = []
        truncateds_ = []
        infos_ = []

        # reset, data_env, behavior_env only differ in reward
        seed = np.random.randint(0, 1000)
        behavior_obs, _ = behavior_env.reset(seed=seed)
        obs, _ = data_env.reset(seed=seed)
        if self.debug:
            print(f"True goal: {obs['desired_goal']}, Sample goal: {behavior_obs['desired_goal']}")

        # Initialize return accumulator, terminated, truncated, info
        achieved_ret = 0
        data_terminated = False
        behavior_terminated = False
        truncated = False
        info = None

        for n_step in range(self.horizon):
            observations_.append(deepcopy(obs['observation']))
            achieved_rets_.append(deepcopy(achieved_ret))
            timesteps_.append(deepcopy(n_step))
            terminateds_.append(data_terminated) # We assume starting point is unfinished
            truncateds_.append(truncated)
            infos_.append(info)

            # if data_terminated: # Reached true goal, don't move, dummy action, change reward to 1
            #     # action = np.zeros(2,)
            #     # reward = 1

            #     # Continue control
            #     pass
                
            # else: 
                # controller uses the 'desired_goal' key of obs to know the goal, not the goal mark on the map
            if behavior_terminated and random_end: # sample goal reached, take no action
                action = behavior_env.action_space.sample()
            else: # still head toward the sample goal
                action = controller.compute_action(behavior_obs)
            # action = controller.compute_action(behavior_obs)

            behavior_obs, _, behavior_terminated, _, _ = behavior_env.step(action)
            if self.debug:
                print(f"Step {n_step}, behavior maze, current pos {behavior_obs['achieved_goal']}, terminated {behavior_terminated}")

            obs, reward, data_terminated, truncated, info = data_env.step(action)
            if self.debug:
                print(f"Step {n_step}, data maze, current pos {obs['achieved_goal']}, terminated {data_terminated}, reward {reward}")

            actions_.append(deepcopy(action))
            rewards_.append(deepcopy(reward))

            # Update return
            achieved_ret += reward

        # Compute returns. Note that achieved_ret is now total return
        total_ret = achieved_ret
        returns_ = [total_ret - achieved for achieved in achieved_rets_]
        trajs.append(Trajectory(observations = observations_, 
                        actions = actions_, 
                        rewards = rewards_, 
                        returns = returns_, 
                        timesteps = timesteps_, 
                        terminated = terminateds_, 
                        truncated = truncateds_, 
                        infos = infos_))
    behavior_env.close()
    data_env.close()

        # if data_terminated:
        #     print(f"Warning: data_env already reached goal, quit sampling immediately")
        #     break
        # if behavior_terminated:
        #     print(f"Behavior env finished")
        #     break



    return trajs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--render',action='store_true', help='Render env')
    parser.add_argument('--maze_config_file', type=str, default='./config/maze1.json')
    parser.add_argument('--horizon', type=int, default=250)
    args = parser.parse_args()
    print(args)


    maze_config = json.load(open(args.maze_config_file, 'r'))
    map = maze_config['map']  

    start = maze_config['start']
    goal = maze_config['goal']

    data_map = set_map_cell(map, start, 'r')
    data_map = set_map_cell(map, goal, 'g')
    data_env = gym.make('PointMaze_UMazeDense-v3', 
                            maze_map= data_map, 
                            continuing_task = False,
                            max_episode_steps=args.horizon)
    
    maze_sampler = MazeSampler(args.horizon, map, start, goal, debug = args.debug, render = args.render)
    avg_expert_ret = maze_sampler.get_expert_return(repeat=10)
    print(f"Averaged expert return is {avg_expert_ret}")


    # Add a get_true_observation method for Env
    # def get_true_observation(obs):
    #     '''
    #     obs, obs received from pointmaze Env
    #     '''
    #     return obs['observation']

    # setattr(env, 'get_true_observation', get_true_observation)