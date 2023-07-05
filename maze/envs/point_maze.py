'''
Point maze environment
'''

from maze.envs.base import BaseOfflineEnv
import gymnasium as gym
import pickle

class PointMaze(BaseOfflineEnv):
    def __init__(self, data_path, horizon, n_trajs, map_path = None ):
        '''
        data_path: path to dataset
        map_path: If not None: path to point maze description file. (May support default env later)
        horizon: horizon of every trajectory
        n_trajs: number of trajectories for dataset
        '''

        # Create a env_cls, a function that returns the Env class
        if map_path is None:
            env_cls = lambda : gym.make('PointMaze_UMaze-v3')
        else:
            with open(map_path, 'rb') as f:
                maze_map = pickle.load(f) # list(list)
            env_cls = lambda : gym.make('PointMaze_UMaze-v3', maze_map=maze_map)

        super().__init__(data_path, env_cls, data_policy_fn, horizon, n_interactions)

    def get_map(self, map_name: str):
        '''
        Get the map according to map_name
        '''