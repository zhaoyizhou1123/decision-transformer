import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, Optional
import pickle
import collections

class DictDataset(Dataset):
    '''
    From Dict to dataset
    '''
    def __init__(self, dict_dataset: Dict[str, np.ndarray], horizon):
        self.dataset = dict_dataset

        # 'obss' and 'next_obss' key may have different names, store its name
        if 'obss' in self.dataset.keys():
            self.obss_key = 'obss'
            self.next_obss_key = 'next_obss'
        else:
            self.obss_key = 'observations'
            self.next_obss_key = 'next_observations'

        self.horizon = horizon


    def __len__(self):
        return len(self.dataset[self.obss_key])
    
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        '''
        Update: Also train incomplete contexts. Incomplete contexts pad 0.
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - states: Tensor of size [ctx_length, state_space_size]
        - actions: Tensor of size [ctx_length, action_dim], here action is converted to one-hot representation
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: (ctx_length) if single_timestep=False; else (1,), only keep the first timestep
        '''
        obs = self.dataset[self.obss_key][index]
        obs = torch.as_tensor(obs).unsqueeze(0)
        action = self.dataset['actions'][index]
        action = torch.as_tensor(action).unsqueeze(0)
        rtg = self.dataset['rtgs'][index]
        rtg = torch.as_tensor(rtg).reshape(1,1)
        timestep = torch.tensor([index % self.horizon])

        return obs, action, rtg, timestep

class DictQDataset(Dataset):
    '''
    From Dict to Q-learning dataset
    '''
    def __init__(self, dict_dataset: Dict[str, np.ndarray], horizon):
        self.dataset = dict_dataset

        # 'obss' and 'next_obss' key may have different names, store its name
        if 'obss' in self.dataset.keys():
            self.obss_key = 'obss'
            self.next_obss_key = 'next_obss'
        else:
            self.obss_key = 'observations'
            self.next_obss_key = 'next_observations'

        self.horizon = horizon


    def __len__(self):
        return len(self.dataset[self.obss_key])
    
    def __getitem__(self, index) -> Dict[str, np.ndarray]:
        '''
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - state: Tensor of size [state_dim]
        - action: Tensor of size [action_dim], here action is converted to one-hot representation
        - reward: Tensor of size [1]
        - next_state: Tensor of size [state_dim]. The terminal state is fixed to 0, and is added by
                      read_data_reward method
        - timestep: scalar?
        '''
        obs = self.dataset[self.obss_key][index]
        obs = torch.as_tensor(obs)
        action = self.dataset['actions'][index]
        action = torch.as_tensor(action)
        reward = self.dataset['rewards'][index]
        reward = torch.as_tensor(reward).reshape(1)
        timestep = torch.tensor(index % self.horizon)

        if index != len(self) - 1:
            next_obs = torch.as_tensor(self.dataset[self.obss_key][index + 1])
        else:
            next_obs = torch.tensor([0], dtype = torch.float32)

        return obs, action, reward, next_obs, timestep
        


def traj_rtg_datasets(env, input_path: Optional[str] =None, data_path: Optional[str] = None):
    '''
    Download all datasets needed for experiments, and re-combine them as trajectory datasets
    Throw away the last uncompleted trajectory

    Args:
        data_dir: path to store dataset file

    Return:
        dataset: Dict,
        initial_obss: np.ndarray
        max_return: float
    '''
    dataset = env.get_dataset(h5path=input_path)

    N = dataset['rewards'].shape[0] # number of data (s,a,r)
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    # obs_ = []
    # next_obs_ = []
    # action_ = []
    # reward_ = []
    # done_ = []
    # rtg_ = []

    for i in range(N): # Loop through data points

        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
            
        # obs_.append(dataset['observations'][i].astype(np.float32))
        # next_obs_.append(dataset['next_observations'][i].astype(np.float32))
        # action_.append(dataset['actions'][i].astype(np.float32))
        # reward_.append(dataset['rewards'][i].astype(np.float32))
        # done_.append(bool(dataset['terminals'][i]))

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            # Update rtg
            rtg_traj = discount_cumsum(np.array(data_['rewards']))
            episode_data['rtgs'] = rtg_traj
            # rtg_ += rtg_traj

            paths.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    init_obss = np.array([p['observations'][0] for p in paths]).astype(np.float32)

    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    print(f'Number of samples collected: {num_samples}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

    if data_path is not None:
        with open(data_path, 'wb') as f:
            pickle.dump(paths, f)

    # print(f"N={N},len(obs_)={len(obs_)},len(reward_)={len(reward_)},len(rtg_)={len(rtg_)}!")
    # assert len(obs_) == len(rtg_), f"Got {len(obs_)} obss, but {len(rtg_)} rtgs!"

    # Concatenate paths into one dataset
    full_dataset = {}
    for k in ['observations', 'next_observations', 'actions', 'rewards', 'rtgs', 'terminals']:
        full_dataset[k] = np.concatenate([p[k] for p in paths], axis=0)

    return full_dataset, init_obss, np.max(returns)

def discount_cumsum(x: np.ndarray, gamma: float = 1.):
    '''
    Used to calculate rtg for rewards seq (x)
    '''
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum