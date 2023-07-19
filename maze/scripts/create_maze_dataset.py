import __init__

from maze.envs.point_maze import PointMaze
from maze.utils.trajectory import show_trajectory
from maze.utils.maze_utils import set_map_cell
import gymnasium as gym

import argparse
import numpy as np
import json
import os
import pickle

def create_env_dataset(args):
    '''
    Create env and dataset (if not created)
    '''
    maze_config = json.load(open(args.maze_config_file, 'r'))
    map = maze_config['map']  

    start = maze_config['start']
    goal = maze_config['goal']

    alt_start = [3,1]
    mid_point = [3,4]

    alt_goal = [1,7]

    sample_starts = []
    sample_goals = []

    n_trajs = int(1e6) // args.horizon

    repeat = n_trajs // 3

    for i in range(repeat): # a suboptimal path
        sample_starts.append(np.array(start))
        sample_goals.append(np.array(alt_goal))
    for i in range(repeat, repeat * 2): # teach path mid->goal
        sample_starts.append(np.array(alt_start))
        sample_goals.append(np.array(goal))    
    for i in range(repeat * 2, n_trajs): # teach path start->mid
        sample_starts.append(np.array(start))
        sample_goals.append(np.array(mid_point))    

    # print(sample_starts)
    # print(sample_goals)

    print(f"Create point maze")
    point_maze = PointMaze(data_path = args.data_file, 
                        horizon = args.horizon,
                        maze_map = map,
                        start = np.array(start),
                        goal = np.array(goal),
                        sample_starts=sample_starts,
                        sample_goals=sample_goals,
                        debug=args.debug)   
    
    return point_maze

def create_env(args):
    '''
    Create env(if not created)
    '''
    maze_config = json.load(open(args.maze_config_file, 'r'))
    map = maze_config['map']  

    start = maze_config['start']
    goal = maze_config['goal']

    target_map = set_map_cell(map, goal, 'g')
    target_map = set_map_cell(target_map, start, 'r')

    # print(sample_starts)
    # print(sample_goals)
    render_mode = "human" if args.debug else "None"
    # render_mode = None

    env = gym.make('PointMaze_UMazeDense-v3', 
             maze_map = target_map, 
             continuing_task = False,
             max_episode_steps=args.horizon,
             render_mode=render_mode)
    # env = gym.make('PointMaze_UMazeDense-v3', 
    #          maze_map = target_map, 
    #          continuing_task = False,
    #          max_episode_steps=args.horizon)
    
    return env

def load_dataset(data_path):
    '''
    Try to load dataset from daa_path. If fails, return none
    '''
    if data_path is not None and os.path.exists(data_path):
        # print('Dataset file found. Loading existing trajectories.')
        with open(data_path, 'rb') as file:
            dataset = pickle.load(file) # Dataset may not be trajs, might contain other infos
        return dataset
    else:
        return None

# print("Trajectory 0")
# show_trajectory(trajs[0])

# print("Trajectory -1")
# show_trajectory(trajs[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='./dataset/maze_1e6.dat',help='./dataset/maze_1e6.dat')
    parser.add_argument('--horizon', type=int, default=250)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--maze_config_file', type=str, default='./config/maze1.json')

    args = parser.parse_args()
    print(args)

    create_env_dataset(args)

