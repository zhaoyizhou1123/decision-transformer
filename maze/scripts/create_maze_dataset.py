import __init__

from maze.envs.point_maze import PointMaze
from maze.utils.trajectory import show_trajectory

import argparse
import numpy as np

def create_env_dataset(args):
    '''
    Create env and dataset (if not created)
    '''
    map = [[1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,0,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1]]   

    start = [1,1]
    goal = [3,7]

    alt_start = [3,1]
    mid_point = [3,4]

    alt_goal = [1,7]

    sample_starts = []
    sample_goals = []

    n_trajs = 6

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

    point_maze = PointMaze(data_path = args.data_file, 
                        horizon = args.horizon,
                        maze_map = map,
                        start = np.array(start),
                        goal = np.array(goal),
                        sample_starts=sample_starts,
                        sample_goals=sample_goals,
                        debug=args.debug)   
    
    return point_maze
# print("Trajectory 0")
# show_trajectory(trajs[0])

# print("Trajectory -1")
# show_trajectory(trajs[-1])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='./dataset/maze_debug.dat')
    parser.add_argument('--horizon', type=int, default=250)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    print(args)

    create_env_dataset(args)

