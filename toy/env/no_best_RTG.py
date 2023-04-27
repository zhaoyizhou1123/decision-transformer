# import random
# import json
import gym
from gym import spaces
# import pandas as pd
import numpy as np
import torch

class BanditEnv(gym.Env):
    '''A repeat bandit env of horizon H'''
    metadata = {'render.modes': ['human']}
    def __init__(self, horizon):
        '''horizon: int'''
        super(BanditEnv, self).__init__()

        # self.df = df
        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions, 0 for bad, 1 for good
        self.action_space = spaces.Box(
            low=0, high=1, dtype=np.int)

        # States, only 1 state
        self.observation_space = spaces.Box(
            low=0, high=0, dtype=np.int)
        
        self._current_state = np.int(0)
        self._timestep = 1 # current timestep
        self._HORIZON = horizon
        self._return = 0 # total rewards

    def step(self, action):
        '''
        action: {0,1}\n
        Return: next_obs: next state, =0; reward: equals action; done: bool, True if H actions taken 
        '''
        next_obs = np.int(0)
        reward = np.int(action)
        self._return += reward

        done = (self._timestep == self._HORIZON)
        # print(f"self._HORIZON is {self._HORIZON}")
        # print(f"self._timestep is {self._timestep}, done is {done}" )
        self._timestep += 1
        
        return torch.Tensor([next_obs]), reward, done
    
    def reset(self):
        '''Return initial state as torch.Tensor of shape (1,)'''
        self._current_state = np.int(0)
        self._timestep = 1
        self._return = 0
        return torch.Tensor([self._current_state])
        
    def render(self, mode='human', close=False):
        print(f"Current state is {self._current_state}")
        print(f"Time step={self._timestep}")
        print(f"Current total reward is {self._return}")

    def get_state(self):
        '''Return current state as int, for dataset creation'''
        return self._current_state

        
    # def _next_observation(self):
    #     # Get the stock data points for the last 5 days and scale to between 0-1
    #     frame = np.array([
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Open'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'High'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Low'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Close'].values / MAX_SHARE_PRICE,
    #         self.df.loc[self.current_step: self.current_step +
    #                     5, 'Volume'].values / MAX_NUM_SHARES,
    #     ])

    #     # Append additional data and scale each value to between 0-1
    #     obs = np.append(frame, [[
    #         self.balance / MAX_ACCOUNT_BALANCE,
    #         self.max_net_worth / MAX_ACCOUNT_BALANCE,
    #         self.shares_held / MAX_NUM_SHARES,
    #         self.cost_basis / MAX_SHARE_PRICE,
    #         self.total_shares_sold / MAX_NUM_SHARES,
    #         self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
    #     ]], axis=0)

    #     return obs