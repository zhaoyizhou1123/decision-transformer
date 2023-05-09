# import random
# import json
import gym
from gym import spaces
# import pandas as pd
import numpy as np
import torch
# from mingpt.utils import state_hash

class BanditEnv(gym.Env):
    '''A repeat bandit env of horizon H'''
    metadata = {'render.modes': ['human']}
    def __init__(self, horizon, state_hash):
        '''
        - horizon: int
        - state_hash: function | None. If not None, specifies a way to hash states'''
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
        self._state_hash = state_hash

        self._num_action = int(2)

    def step(self, action):
        '''
        action: {0,1}\n
        Return: next_obs: next state, =0; reward: equals action; done: bool, True if H actions taken 
        '''
        next_obs = np.int(0)
        next_obs = self._hash_state(next_obs)

        reward = np.int(action)
        self._return += reward

        done = (self._timestep == self._HORIZON)
        # print(f"self._HORIZON is {self._HORIZON}")
        # print(f"self._timestep is {self._timestep}, done is {done}" )
        self._timestep += 1
        
        return torch.Tensor([next_obs]), reward, done
    
    def reset(self):
        '''Return initial state as torch.Tensor of shape (1,)'''
        current_state = np.int(0)
        self._current_state = self._hash_state(current_state)

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
    
    def get_num_action(self):
        '''
        Return the number of possible actions. For initializing empty action Tensor
        '''
        return self._num_action
    
    def _hash_state(self, state):
        '''
        Return the hashed state according to self._state_hash \n
        Input: state, int
        Output: state, (scalar)
        '''
        if self._state_hash is not None:
            state = self._state_hash(state)
        return state

    
class BanditEnvReverse(gym.Env):
    '''Similar as BanditEnv, but reward 0 for action 1, 1 for action 0'''
    metadata = {'render.modes': ['human']}
    def __init__(self, horizon):
        '''horizon: int'''
        super(BanditEnvReverse, self).__init__()

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
        reward = np.int(1-action)
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

class BanditEnvOneHot(gym.Env):
    '''Action is represented in one-hot'''
    metadata = {'render.modes': ['human']}
    def __init__(self, horizon):
        '''horizon: int'''
        super().__init__()

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

        self._num_action = int(2)
        self._state_dim = int(1)

    def step(self, action):
        '''
        action: Tensor [1,0] (0) or [0,1] (1)
        Return: next_obs: next state, =0; reward: equals action; done: bool, True if H actions taken 
        '''
        if torch.equal(action, torch.Tensor([1,0])):
            action_decode = 0
        elif torch.equal(action, torch.Tensor([0,1])):
            action_decode = 1
        else:
            raise Exception(f"BanditEnvOneHot: Invalid action representation {action}")
        next_obs = np.int(0)
        reward = np.int(action_decode)
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
    
    def get_dims(self):
        '''
        Return state and action dimensions, can replace get_num_action
        '''
        return self._state_dim, self._num_action
    
    def get_num_action(self):
        '''
        Return the number of possible actions. For initializing empty action Tensor
        '''
        return self._num_action       