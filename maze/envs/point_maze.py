'''
Point maze environment
'''

from maze.envs.base import BaseOfflineEnv
from maze.utils.maze_utils import set_map_cell
from maze.samplers.maze_sampler import MazeSampler
import gymnasium as gym
import pickle
from gymnasium.spaces import Box
from typing import Dict
import numpy as np

def get_true_obs(obs):
    '''
    obs, obs received from pointmaze Env
    '''
    # print(obs)
    return obs['observation']

class ObsWrapper(gym.Wrapper):
    '''
    Wrap pick place environment to return desired obs
    '''
    def __init__(self, env):
        super().__init__(env)
        # Get observation space
        tmp_obs, _ = env.reset()

        tmp_true_obs = get_true_obs(tmp_obs)
        low = env.observation_space['observation'].low[0]
        high = env.observation_space['observation'].high[0]
        self.observation_space = Box(shape = tmp_true_obs.shape, low = low, high = high)

    # def observation(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
    #     return get_true_obs(observation)

    def step(self, a):
        next_s, reward, terminated, _, info = self.env.step(a)
        return get_true_obs(next_s),reward, terminated, info

    def reset(self, seed = None):
        state, _ = self.env.reset(seed = seed)
        return get_true_obs(state)

class PointMaze(BaseOfflineEnv):
    def __init__(self, data_path, horizon, maze_map, start, goal, 
                 sample_args,
                 debug=False, render=False, use_wrapper = False):
        '''
        data_path: path to dataset
        maze_map: list(list), basic map of the game, only specifies 0, 1. R and G are specified by start and goal.
        start, array (2,), int, the start point of the game we want to learn
        goal, array (2,), int, the goal point of the game we want to learn
        horizon: horizon of every trajectory
        n_trajs: number of trajectories for dataset
        sample_starts: list, list of starts for sampling
        sample_goals: list, list of goals for sampling
        '''

        # Create a env_cls, a function that returns the Env class
        # if map_path is None:
        #     env_cls = lambda : gym.make('PointMaze_UMaze-v3')
        # else:
        #     with open(map_path, 'rb') as f:
        #         maze_map = pickle.load(f) # list(list)

        self.MAZE_MAP = maze_map
        target_map = set_map_cell(self.MAZE_MAP, goal, 'g')
        target_map = set_map_cell(target_map, start, 'r')

        render_mode = "human" if render else None
        # render_mode = None
        if not use_wrapper:
            env_cls = lambda : gym.make('PointMaze_UMazeDense-v3', 
                                        maze_map = target_map, 
                                        continuing_task = False,
                                        max_episode_steps=self.horizon,
                                        render_mode=render_mode)
        else:
            print(f"Use wrapper")
            env_cls = lambda : ObsWrapper(gym.make('PointMaze_UMazeDense-v3', 
                                        maze_map = target_map, 
                                        continuing_task = False,
                                        max_episode_steps=self.horizon,
                                        render_mode=render_mode))         
        
        sampler = MazeSampler(horizon=horizon,
                              maze_map=self.MAZE_MAP,
                              target_start=start,
                              target_goal=goal,
                              debug=debug,
                              render=render)
        
        # sample_args = {'starts': sample_starts, 'goals': sample_goals, 'repeats': sample_repeats, 'randoms': sample_end_randoms}
        
        super().__init__(data_path, env_cls, horizon,
                         sampler = sampler, sample_args=sample_args)


    # def get_map(self, map_name: str):
    #     '''
    #     Get the map according to map_name
    #     '''