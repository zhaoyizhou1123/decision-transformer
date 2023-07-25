import os
import sys

# print(f"test: {os.getcwd()}")
sys.path.append(os.getcwd()+"/../..")

import pickle
from maze.utils.trajectory import show_trajectory

data_file = "maze2.dat"

with open(data_file, "rb") as f:
    trajs, horizon, map, start, goal = pickle.load(f)
print(f"Horizon: {horizon}")
print(f"Map: {map}")
print(f"Start: {start}; Goal: {goal}")
print(f"Trajectory number: {len(trajs)}")
# for idx, traj in enumerate(trajs):
#     print(f'--------------------\nTrajectory {idx}')
#     show_trajectory(traj, timesteps = [0])