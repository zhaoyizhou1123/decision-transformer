import os
import sys

# print(f"test: {os.getcwd()}")
sys.path.append(os.getcwd()+"/../..")

import pickle
from maze.utils.trajectory import show_trajectory

data_file = "maze.dat"

with open(data_file, "rb") as f:
    trajs, map, start, goal = pickle.load(f)
print(f"Map: {map}")
print(f"Start: {start}; Goal: {goal}")
for idx, traj in enumerate(trajs):
    print(f'--------------------\nTrajectory {idx}')
    show_trajectory(traj, timesteps = [0])