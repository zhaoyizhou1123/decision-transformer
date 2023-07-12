import os
import sys

# print(f"test: {os.getcwd()}")
sys.path.append(os.getcwd()+"/..")

import gymnasium as gym
import numpy as np
from samplers.maze_sampler import MazeSampler
# import d4rl

map = [[1,1,1,1,1,1,1,1,1],
       [1,0,0,0,1,0,0,0,1],
       [1,1,1,0,0,0,1,1,1],
       [1,0,0,0,1,0,0,0,1],
       [1,1,1,1,1,1,1,1,1]]   

# # start = [1,1]
# # alt_goal = [3,7]

# # target_goal = [3,1]

# map = [[1,1,1,1,1,1],
#        [1,0,0,0,0,1],
#        [1,0,0,0,0,1],
#        [1,0,0,0,0,1],
#        [1,1,1,1,1,1]]

# # map2 = [[1,1,1,1,1,1,1,1,1],
# #        [1,'r',0,0,1,0,0,0,1],
# #        [1,1,1,0,0,0,1,1,1],
# #        [1,0,0,0,1,0,0,'g',1],
# #        [1,1,1,1,1,1,1,1,1]] 


# start = [1,1]
# target_goal = [1,3]
# alt_goal = [3,1]

# sampler = MazeSampler(horizon=400,
#                       maze_map=map,
#                       target_goal=np.array(target_goal),
#                       debug=True)

# sampler._collect_single_traj(start=np.array(start), goal=np.array(alt_goal))


# for i in range(4):
#     map = [[1, 1, 1, 1, 1, 1],
#        [1, 'r', 0, 0, 'g', 1],
#        [1, 1, 1, 1, 1, 1]]
#     # options={'goal_cell':goal, 'reset_cell': start}
#     # print(options["goal_cell"][1], options["goal_cell"][0])
#     env = gym.make('PointMaze_UMazeDense-v3', maze_map = map)
#     obs, _ = env.reset()
#     print(obs['observation'], obs['desired_goal'])
    # print(info)
# print(env.observation_space)

env = gym.make('PointMaze_UMazeDense-v3', maze_map = map)
print(env.observation_space)

