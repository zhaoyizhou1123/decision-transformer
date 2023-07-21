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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',action='store_true', help='Print debuuging info if true')
    parser.add_argument('--render',action='store_true', help='Render env')
    parser.add_argument('--maze_config_file', type=str, default='./config/maze2.json')
    parser.add_argument('--horizon', type=int, default=300)
    args = parser.parse_args()
    print(args)


    maze_config = json.load(open(args.maze_config_file, 'r'))
    maze = maze_config["maze"]
    map = maze['map']  

    start = maze['start']
    # start = np.array([-3,-1])
    goal = maze['goal']

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