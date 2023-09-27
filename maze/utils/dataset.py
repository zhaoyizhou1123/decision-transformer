'''
Create the Dataset Class from csv file
'''

import torch
from torch.utils.data import Dataset
# import pandas as pd
import numpy as np
from typing import List
from maze.utils.trajectory import Trajectory

class BanditReturnDataset(Dataset):
    '''Son of the pytorch Dataset class'''

    def __init__(self, states, block_size, actions, rtgs, timesteps, single_timestep = False):    
        '''
        single_timestep: bool. If true, timestep only keep initial step; Else (ctx,)
        '''    
        self.block_size = block_size # args.context_length*3
        self.vocab_size = 2 # Specifies number of actions. The actions are represented s {0,1,2,...}
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        self.data = states #obss
        self.actions = actions # np.array (num_trajectories, horizon)
        # self.done_idxs = done_idxs # What is this for?
        self.rtgs = rtgs
        self.timesteps = timesteps # (trajectory_num,horizon+1)
        self._trajectory_num = states.shape[0] # number of trajectories in dataset
        self._horizon = states.shape[1]
        self.single_timestep = single_timestep

        # number of trajectories should match
        assert self._trajectory_num == actions.shape[0] 
        assert self._trajectory_num == rtgs.shape[0] 
        assert self._trajectory_num == timesteps.shape[0] 
    
    def __len__(self):
        return self._trajectory_num * self._horizon
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        '''
        Update: Also train incomplete contexts. Incomplete contexts pad 0.
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - states: Tensor of size [ctx_length, state_space_size]
        - actions: Tensor of size [ctx_length, action_dim], here action is converted to one-hot representation
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: (ctx_length) if single_timestep=False; else (1,), only keep the first timestep
        '''
        # block_size = self.block_size // 3  # "//" means [5/3], here approx context length
        # done_idx = idx + block_size
        # for i in self.done_idxs:
        #     if i > idx: # first done_idx greater than idx
        #         done_idx = min(int(i), done_idx)
        #         break
        # idx = done_idx - block_size
        # states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.

        ctx = self.block_size // 3 # context length
        data_num_per_trajectory = self._horizon # number of data one trajectory provides
        trajectory_idx = idx // data_num_per_trajectory # which trajectory to read, row
        res_idx = idx - trajectory_idx * data_num_per_trajectory # column index to read
        
        assert res_idx < self._horizon, idx
        assert trajectory_idx < self._trajectory_num, idx

        # Test whether it is full context length
        if res_idx - ctx + 1 < 0:
            start_idx = 0
            pad_len = ctx - res_idx - 1 # number of zeros to pad
        else:
            start_idx = res_idx - ctx + 1
            pad_len = 0


        states_slice = self.data[trajectory_idx, start_idx : res_idx + 1, :]
        actions_slice = self.actions[trajectory_idx, start_idx : res_idx + 1, :]
        rtgs_slice = self.rtgs[trajectory_idx, start_idx : res_idx + 1, :]

        # pad 0
        states_slice = torch.cat([torch.zeros(pad_len, states_slice.shape[-1]), states_slice], dim = 0)
        actions_slice = torch.cat([torch.zeros(pad_len, actions_slice.shape[-1]), actions_slice], dim = 0)
        rtgs_slice = torch.cat([torch.zeros(pad_len, rtgs_slice.shape[-1]), rtgs_slice], dim = 0)

        if self.single_timestep: # take the last step
            timesteps_slice = self.timesteps[trajectory_idx, res_idx : res_idx + 1] # (1,)
        else: 
            timesteps_slice = self.timesteps[trajectory_idx, start_idx : res_idx + 1] #(real_ctx_len, )
            timesteps_slice = torch.cat([torch.zeros(pad_len), timesteps_slice], dim = 0)

        
        
        # print(f"Size of output: {states_slice.shape}, {actions_slice.shape}, {rtgs_slice.shape}, {timesteps_slice.shape}")
        # print(f"Dataset actions_slice: {actions_slice.shape}")
        return states_slice, actions_slice, rtgs_slice, timesteps_slice
    
    def getitem(self, idx):
        states_slice, actions_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
        return states_slice, actions_slice, rtgs_slice, timesteps_slice
    
class BanditRewardDataset(Dataset):
    '''
    Son of the pytorch Dataset class \n
    Return (s,a,r,s',t)
    '''

    def __init__(self, states, actions, rewards, timesteps, state_hash=None, action_hash=None, time_order = False):       
        '''
        state_hash: function | None. If not None, specifies a way to hash states \n
        time_order: If true, get data in the order of t= H-1,H-2,...,0, used for ordered training. Should be true when shuffle=False
        ''' 
        # self.block_size = block_size # args.context_length*3
        self.vocab_size = 2 # Specifies number of actions. The actions are represented s {0,1,2,...}
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        self.states = states #obss (num_trajectories, horizon+1)
        self.actions = actions # np.array (num_trajectories, horizon)
        # self.done_idxs = done_idxs # What is this for?
        self.rewards = rewards
        self.timesteps = timesteps # (trajectory_num,horizon+1)

        self._state_hash = state_hash
        self._action_hash = action_hash

        self._trajectory_num = states.shape[0] # number of trajectories in dataset
        self._horizon = actions.shape[1]

        self._time_order = time_order

        # number of trajectories should match
        assert self._trajectory_num == actions.shape[0] 
        assert self._trajectory_num == rewards.shape[0] 
        assert self._trajectory_num == timesteps.shape[0] 
    
    def __len__(self):
        return self._trajectory_num*self._horizon
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
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
        # block_size = self.block_size // 3  # "//" means [5/3], here approx context length
        # done_idx = idx + block_size
        # for i in self.done_idxs:
        #     if i > idx: # first done_idx greater than idx
        #         done_idx = min(int(i), done_idx)
        #         break
        # idx = done_idx - block_size
        # states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.

        # ctx = self.block_size // 3 # context length
        if not self._time_order:
            data_num_per_trajectory = self._horizon  # number of data one trajectory provides
            trajectory_idx = idx // data_num_per_trajectory # which trajectory to read, row
            res_idx = idx - trajectory_idx * data_num_per_trajectory # column index to read
        else: # (0, H-1), (1,H-1), ..., (num_traj-1,H-1),(0,H-2),...,(num_traj-1, 0), in this order
            res_idx = self._horizon - 1 - idx // self._trajectory_num
            trajectory_idx = idx % self._trajectory_num
        
        assert res_idx < self._horizon, idx
        assert trajectory_idx < self._trajectory_num, idx
        
        state = self.states[trajectory_idx, res_idx, :]
        state = self._hash_state(state) # hash the state

        action = self.actions[trajectory_idx, res_idx, :]
        action = self._hash_action(action)

        reward = self.rewards[trajectory_idx, res_idx, :]

        next_state = self.states[trajectory_idx, res_idx+1, :]
        next_state = self._hash_state(next_state)

        timestep= self.timesteps[trajectory_idx, res_idx]
        
        # print(f"Size of output: {states_slice.shape}, {actions_slice.shape}, {rtgs_slice.shape}, {timesteps_slice.shape}")
        # print(f"Dataset actions_slice: {actions_slice.shape}")
        return state, action, reward, next_state, timestep
    
    # def getitem(self, idx):
    #     states_slice, actions_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
    #     return states_slice, actions_slice, rtgs_slice, timesteps_slice

    def _hash_state(self, state):
        '''
        Return the hashed state according to self._state_hash \n
        Input: state, Tensor (1)
        Output: state, Tensor (1)
        '''
        if self._state_hash is not None:
            hashed_state = self._state_hash(state)
            return hashed_state
        else:
            return state
        
    def _hash_action(self, action):
        '''
        Return the hashed action according to self._action_hash \n
        Input: action, Tensor (1)
        Output: action, Tensor (1)
        '''
        if self._action_hash is not None:
            hashed_action = self._action_hash(action)
            return hashed_action
        else:
            return action
    
class TrajCtxDataset(Dataset):
    '''
    Son of the pytorch Dataset class
    Provides context length, no next state.
    '''

    def __init__(self, trajs, ctx = 1, single_timestep = False, keep_ctx = True, with_mask=False, state_normalize=False):    
        '''
        trajs: list(traj), namedtuple "observations", "actions", "rewards", "returns", "timesteps", "terminated", "truncated", "infos" \n
        single_timestep: bool. If true, timestep only keep initial step; Else (ctx,) \n
        keep_ctx: If False, ctx must be set 1, and we will not keep ctx dimension.
        with_mask: If true, also return attention mask. For DT
        state_normalize: If true, normalize states
        Note: Each traj must have same number of timesteps
        '''    
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        # self.obs = []
        # self.actions = []
        # self.rewards = []
        # self.rts = []
        # self.timesteps = []
        # for traj in trajs:
        #     self.obs += traj.observations #obss
        #     self.actions += traj.actions # np.array (num_trajectories, horizon)
        #     # self.done_idxs = done_idxs # What is this for?
        #     self.rewards += traj.rewards
        #     self.rtgs += traj.returns
        #     self.timesteps += traj.timesteps # (trajectory_num,horizon+1)
        #     # self._trajectory_num = states.shape[0] # number of trajectories in dataset
        #     # self._horizon = states.shape[1]
        self._trajs = trajs
        self._trajectory_num = len(self._trajs)
        self._horizon = len(self._trajs[0].observations)
        self.keep_ctx = keep_ctx
        self.with_mask = with_mask

        if not keep_ctx:
            assert ctx == 1, f"When keep_ctx = False, ctx must be 1"

        self.ctx = ctx
        self.single_timestep = single_timestep

        self.state_normalize = state_normalize

        if state_normalize:
            states_list = []
            for traj in trajs:
                states_list += traj.observations
            states = np.concatenate(states_list, axis = 0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        else:
            self.state_mean = 0
            self.state_std = 1

        # number of trajectories should match
        # assert self._trajectory_num == actions.shape[0] 
        # assert self._trajectory_num == rtgs.shape[0] 
        # assert self._trajectory_num == timesteps.shape[0] 
    
    def __len__(self):
        return self._trajectory_num * self._horizon
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        '''
        Update: Also train incomplete contexts. Incomplete contexts pad 0.
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - states: Tensor of size [ctx_length, state_space_size]
        - actions: Tensor of size [ctx_length, action_dim], here action is converted to one-hot representation
        - rewards: Tensor of size [ctx_length, 1]
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: (ctx_length) if single_timestep=False; else (1,), only keep the first timestep
        Note: if keep_ctx = False, all returns above will remove the first dim. In particular, timesteps becomes scalar.
        '''
        # block_size = self.block_size // 3  # "//" means [5/3], here approx context length
        # done_idx = idx + block_size
        # for i in self.done_idxs:
        #     if i > idx: # first done_idx greater than idx
        #         done_idx = min(int(i), done_idx)
        #         break
        # idx = done_idx - block_size
        # states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.

        ctx = self.ctx # context length
        data_num_per_trajectory = self._horizon # number of data one trajectory provides
        trajectory_idx = idx // data_num_per_trajectory # which trajectory to read, row
        res_idx = idx - trajectory_idx * data_num_per_trajectory # column index to read
        
        assert res_idx < self._horizon, idx
        assert trajectory_idx < self._trajectory_num, idx

        # Test whether it is full context length
        if res_idx - ctx + 1 < 0:
            start_idx = 0
            pad_len = ctx - res_idx - 1 # number of zeros to pad
        else:
            start_idx = res_idx - ctx + 1
            pad_len = 0

        traj = self._trajs[trajectory_idx]
        # print(f"Obs shape: {np.array(traj.observations).shape}")
        # states_slice = self.data[trajectory_idx, start_idx : res_idx + 1, :]
        # actions_slice = self.actions[trajectory_idx, start_idx : res_idx + 1, :]
        # rtgs_slice = self.rtgs[trajectory_idx, start_idx : res_idx + 1, :]
        states_slice = torch.from_numpy(np.array(traj.observations)[start_idx : res_idx + 1, :])
        states_slice = (states_slice - self.state_mean) / self.state_std

        # print(f"start: {start_idx}, end: {res_idx}")
        # print(f"Before cat: {states_slice.shape}")

        actions_slice = torch.from_numpy(np.array(traj.actions)[start_idx : res_idx + 1, :])
        rewards_slice = torch.from_numpy(np.array(traj.rewards)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)
        rtgs_slice = torch.from_numpy(np.array(traj.returns)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)

        # pad 0
        states_slice = torch.cat([torch.zeros(pad_len, states_slice.shape[-1]), states_slice], dim = 0)
        actions_slice = torch.cat([torch.zeros(pad_len, actions_slice.shape[-1]), actions_slice], dim = 0)
        rewards_slice = torch.cat([torch.zeros(pad_len, rewards_slice.shape[-1]), rtgs_slice], dim = 0)
        rtgs_slice = torch.cat([torch.zeros(pad_len, rtgs_slice.shape[-1]), rtgs_slice], dim = 0)

        if self.single_timestep: # take the last step
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[res_idx : res_idx + 1]) # (1,)
        else: 
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[start_idx : res_idx + 1]) #(real_ctx_len, )
            timesteps_slice = torch.cat([torch.zeros(pad_len), timesteps_slice], dim = 0)

        # print(f"Size of output: {states_slice.shape}, {actions_slice.shape}, {rtgs_slice.shape}, {timesteps_slice.shape}")
        # print(f"Dataset actions_slice: {actions_slice.shape}")
        if not self.keep_ctx:
            states_slice = states_slice[0,:]
            actions_slice = actions_slice[0,:]
            rewards_slice = rewards_slice[0,:]
            rtgs_slice = rtgs_slice[0,:]
            timesteps_slice = timesteps_slice[0]

        assert states_slice.shape[0] != 0, f"{idx}, {states_slice.shape}"
        if self.with_mask:
            attn_mask = torch.cat([torch.zeros((pad_len)), torch.ones((ctx-pad_len))], dim=-1)
            # print(f"Dataset: Successfully get index {idx}")
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask
        else:
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice
    
    def getitem(self, idx):
        if self.with_mask:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask
        else:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice           

    
    def get_max_return(self):
        traj_rets = [traj.returns[0] for traj in self._trajs]
        return max(traj_rets)
    
    def get_normalize_coef(self):
        '''
        Get state normalization mean and std
        '''
        return self.state_mean, self.state_std

    
class TrajNextObsDataset(Dataset):
    '''
    Son of the pytorch Dataset class
    Don't provide ctx, provide next state
    '''

    def __init__(self, trajs):    
        '''
        trajs: list(traj), namedtuple "observations", "actions", "rewards", "returns", "timesteps", "terminated", "truncated", "infos" \n
        Note: Each traj must have same number of timesteps
        '''    
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        # self.obs = []
        # self.actions = []
        # self.rewards = []
        # self.rts = []
        # self.timesteps = []
        # for traj in trajs:
        #     self.obs += traj.observations #obss
        #     self.actions += traj.actions # np.array (num_trajectories, horizon)
        #     # self.done_idxs = done_idxs # What is this for?
        #     self.rewards += traj.rewards
        #     self.rtgs += traj.returns
        #     self.timesteps += traj.timesteps # (trajectory_num,horizon+1)
        #     # self._trajectory_num = states.shape[0] # number of trajectories in dataset
        #     # self._horizon = states.shape[1]
        self._trajs = trajs
        self._trajectory_num = len(self._trajs)
        self._horizon = len(self._trajs[0].observations)

        # Lazy way, use the code of TrajCtxDataset
        # self.keep_ctx = False
        # self.ctx = 1
        # self.single_timestep = single_timestep

        # number of trajectories should match
        # assert self._trajectory_num == actions.shape[0] 
        # assert self._trajectory_num == rtgs.shape[0] 
        # assert self._trajectory_num == timesteps.shape[0] 
    
    def __len__(self):
        return self._trajectory_num * self._horizon
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        '''
        Update: Also train incomplete contexts. Incomplete contexts pad 0.
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - states: Tensor of size [state_dim]
        - actions: Tensor of size [action_dim], here action is converted to one-hot representation
        - rewards: Tensor of size [1]
        - rtgs: Tensor of size [1]
        - timesteps: scalar
        - next_state: [state_dim]. If reach the last step, filled with 0, should not be used to training.
        Note: if keep_ctx = False, all returns above will remove the first dim. In particular, timesteps becomes scalar.
        '''
        data_num_per_trajectory = self._horizon # number of data one trajectory provides
        trajectory_idx = idx // data_num_per_trajectory # which trajectory to read, row
        res_idx = idx - trajectory_idx * data_num_per_trajectory # column index to read
        
        assert res_idx < self._horizon, idx
        assert trajectory_idx < self._trajectory_num, idx

        # Test whether it is full context length
        # if res_idx - ctx + 1 < 0:
        #     start_idx = 0
        #     pad_len = ctx - res_idx - 1 # number of zeros to pad
        # else:
        #     start_idx = res_idx - ctx + 1
        #     pad_len = 0

        traj = self._trajs[trajectory_idx]

        # Construct next
        # states_slice = self.data[trajectory_idx, start_idx : res_idx + 1, :]
        # actions_slice = self.actions[trajectory_idx, start_idx : res_idx + 1, :]
        # rtgs_slice = self.rtgs[trajectory_idx, start_idx : res_idx + 1, :]
        state = torch.from_numpy(np.array(traj.observations)[res_idx, :]) # (state_dim)
        action = torch.from_numpy(np.array(traj.actions)[res_idx, :])
        reward = torch.tensor(np.array(traj.rewards)[res_idx]).unsqueeze(-1) # (1)
        rtg = torch.tensor(np.array(traj.returns)[res_idx]).unsqueeze(-1) # (1)
        timestep = torch.tensor(np.array(traj.timesteps)[res_idx]) # scalar

        if res_idx < self._horizon - 1:
            next_state = torch.from_numpy(np.array(traj.observations)[res_idx + 1 ,:])
        else: # Reach end of trajectory, pad with 0
            next_state = torch.zeros_like(state) # (state_dim)

        return state, action, reward, rtg, timestep, next_state
    
    def getitem(self, idx):
        states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
        return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice
    
    def get_max_return(self):
        traj_rets = [traj.returns[0] for traj in self._trajs]
        return max(traj_rets)
    

class ObsActDataset(Dataset):
    '''
    From Chuning, for diffusion policy training
    '''
    def __init__(self, trajs: List):
        '''
        trajs: list(traj), namedtuple "observations", "actions", "rewards", "returns", "timesteps", "terminated", "truncated", "infos" \n
        '''
        obs_trajs = [np.array(traj.observations) for traj in trajs]
        obs_trajs = np.array(obs_trajs) # np.array (num_trajs,horizon,obs_dim)

        act_trajs = [traj.actions for traj in trajs]
        act_trajs = np.array(act_trajs) # np.array (num_trajs,horizon,act_dim)
        
        self.observations = obs_trajs.reshape(
            np.prod(obs_trajs.shape[:2]), obs_trajs.shape[2]
        )
        self.actions = act_trajs.reshape(
            np.prod(act_trajs.shape[:2]), act_trajs.shape[2]
        )

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs_sample = self.observations[idx]
        act_sample = self.actions[idx]
        return dict(obs=obs_sample, action=act_sample)

class TrajCtxMixSampler:
    '''
    Sample trajs from mixed dataset
    '''
    def __init__(self, datasets: List[List[Trajectory]], weights: List[float], ctx: int) -> None:
        assert len(datasets) == len(weights), f"Datasets {len(datasets)} and weights {len(weights)} must be of same length!"
        assert all(w>=0 for w in weights) and sum(weights)==1, f"Weights must be valid prob. distribution!"
        self.datasets = datasets
        self.weights = weights
        self.ctx = ctx
    def get_batch_traj(self, batch_size: int, with_mask = False):
        '''
        Get a batch from several datasets using weighted sampling.
        ctx: Context length. Pad 0
        with_mask: If True, also returns mask for DT training.
        '''
        datasets = self.datasets
        weights = self.weights
        ctx = self.ctx

        num_samples = [int(batch_size * w) for w in weights]
        num_samples[-1] = batch_size - sum(num_samples[:-1]) # Make num_samples sum up to batch_size

        batch_s, batch_a, batch_r, batch_rtg, batch_t, batch_mask = [], [], [], [], [], []
        for idx, dataset in enumerate(datasets):
            # dataset is a collection of trajs
            num_sample = num_samples[idx]
            num_trajs = len(dataset)
            horizon = len(dataset[0].observations)
            # Get the data indexs for one dataset
            batch_inds = np.random.choice(
                np.arange(num_trajs * horizon),
                size=num_sample,
                replace=True,
            )
            for i in range(num_sample):
                traj_idx = batch_inds[i] // horizon # which trajectory to read, row
                res_idx = batch_inds[i] - traj_idx * horizon # column index to read
                traj = dataset[traj_idx]

                # Test whether it is full context length
                if res_idx - ctx + 1 < 0:
                    start_idx = 0
                    pad_len = ctx - res_idx - 1 # number of zeros to pad
                else:
                    start_idx = res_idx - ctx + 1
                    pad_len = 0

                states_slice = torch.from_numpy(np.array(traj.observations)[start_idx : res_idx + 1, :])

                actions_slice = torch.from_numpy(np.array(traj.actions)[start_idx : res_idx + 1, :])
                rewards_slice = torch.from_numpy(np.array(traj.rewards)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)
                rtgs_slice = torch.from_numpy(np.array(traj.returns)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)

                # pad 0
                states_slice = torch.cat([torch.zeros(pad_len, states_slice.shape[-1]), states_slice], dim = 0)
                actions_slice = torch.cat([torch.zeros(pad_len, actions_slice.shape[-1]), actions_slice], dim = 0)
                rewards_slice = torch.cat([torch.zeros(pad_len, rewards_slice.shape[-1]), rtgs_slice], dim = 0)
                rtgs_slice = torch.cat([torch.zeros(pad_len, rtgs_slice.shape[-1]), rtgs_slice], dim = 0)

                timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[start_idx : res_idx + 1]) #(real_ctx_len, )
                timesteps_slice = torch.cat([torch.zeros(pad_len), timesteps_slice], dim = 0)

                batch_s.append(states_slice.unsqueeze(0))
                batch_a.append(actions_slice.unsqueeze(0))
                batch_r.append(rewards_slice.unsqueeze(0))
                batch_rtg.append(rtgs_slice.unsqueeze(0))
                batch_t.append(timesteps_slice.unsqueeze(0))

                if with_mask:
                    attn_mask = torch.cat([torch.zeros((pad_len)), torch.ones((ctx-pad_len))], dim=-1)
                    batch_mask.append(attn_mask.unsqueeze(0))
        # print(batch_s[0].shape, batch_s[1].shape)
        batch_s = torch.cat(batch_s, dim=0)
        batch_a = torch.cat(batch_a, dim=0)
        batch_r = torch.cat(batch_r, dim=0)
        batch_rtg = torch.cat(batch_rtg, dim=0)
        batch_t = torch.cat(batch_t, dim=0)
        if with_mask:
            batch_mask.append(batch_mask)
            return batch_s, batch_a, batch_r, batch_rtg, batch_t, batch_mask
        else:
            return batch_s, batch_a, batch_r, batch_rtg, batch_t
        
class TrajCtxWeightedDataset(Dataset):
    '''
    Son of the pytorch Dataset class
    Provides context length, no next state.
    Adds weight for each traj.
    '''

    def __init__(self, trajs: List[Trajectory], weights: List[float], ctx = 1, single_timestep = False, keep_ctx = True, with_mask=False, state_normalize=False):    
        '''
        trajs: list(traj), namedtuple "observations", "actions", "rewards", "returns", "timesteps", "terminated", "truncated", "infos" \n
        weights: list(float). Weight of trajs
        single_timestep: bool. If true, timestep only keep initial step; Else (ctx,) \n
        keep_ctx: If False, ctx must be set 1, and we will not keep ctx dimension.
        with_mask: If true, also return attention mask. For DT
        state_normalize: If true, normalize states
        Note: Each traj must have same number of timesteps
        '''    
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        # self.obs = []
        # self.actions = []
        # self.rewards = []
        # self.rts = []
        # self.timesteps = []
        # for traj in trajs:
        #     self.obs += traj.observations #obss
        #     self.actions += traj.actions # np.array (num_trajectories, horizon)
        #     # self.done_idxs = done_idxs # What is this for?
        #     self.rewards += traj.rewards
        #     self.rtgs += traj.returns
        #     self.timesteps += traj.timesteps # (trajectory_num,horizon+1)
        #     # self._trajectory_num = states.shape[0] # number of trajectories in dataset
        #     # self._horizon = states.shape[1]
        assert len(trajs) == len(weights)
        self._trajs = trajs
        self._weights = weights
        self._trajectory_num = len(self._trajs)
        self._horizon = len(self._trajs[0].observations)
        self.keep_ctx = keep_ctx
        self.with_mask = with_mask

        if not keep_ctx:
            assert ctx == 1, f"When keep_ctx = False, ctx must be 1"

        self.ctx = ctx
        self.single_timestep = single_timestep

        self.state_normalize = state_normalize

        if state_normalize:
            states_list = []
            for traj in trajs:
                states_list += traj.observations
            states = np.concatenate(states_list, axis = 0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        else:
            self.state_mean = 0
            self.state_std = 1

        # number of trajectories should match
        # assert self._trajectory_num == actions.shape[0] 
        # assert self._trajectory_num == rtgs.shape[0] 
        # assert self._trajectory_num == timesteps.shape[0] 
    
    def __len__(self):
        return self._trajectory_num * self._horizon
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        '''
        Update: Also train incomplete contexts. Incomplete contexts pad 0.
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length, and its weight \n
        - states: Tensor of size [ctx_length, state_space_size]
        - actions: Tensor of size [ctx_length, action_dim], here action is converted to one-hot representation
        - rewards: Tensor of size [ctx_length, 1]
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: (ctx_length) if single_timestep=False; else (1,), only keep the first timestep
        - weight: float
        Note: if keep_ctx = False, all returns above will remove the first dim. In particular, timesteps becomes scalar.
        '''
        # block_size = self.block_size // 3  # "//" means [5/3], here approx context length
        # done_idx = idx + block_size
        # for i in self.done_idxs:
        #     if i > idx: # first done_idx greater than idx
        #         done_idx = min(int(i), done_idx)
        #         break
        # idx = done_idx - block_size
        # states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.

        ctx = self.ctx # context length
        data_num_per_trajectory = self._horizon # number of data one trajectory provides
        trajectory_idx = idx // data_num_per_trajectory # which trajectory to read, row
        res_idx = idx - trajectory_idx * data_num_per_trajectory # column index to read
        
        assert res_idx < self._horizon, idx
        assert trajectory_idx < self._trajectory_num, idx

        # Test whether it is full context length
        if res_idx - ctx + 1 < 0:
            start_idx = 0
            pad_len = ctx - res_idx - 1 # number of zeros to pad
        else:
            start_idx = res_idx - ctx + 1
            pad_len = 0

        traj = self._trajs[trajectory_idx]
        weight = self._weights[trajectory_idx] # Get weight for this traj
        # states_slice = self.data[trajectory_idx, start_idx : res_idx + 1, :]
        # actions_slice = self.actions[trajectory_idx, start_idx : res_idx + 1, :]
        # rtgs_slice = self.rtgs[trajectory_idx, start_idx : res_idx + 1, :]
        states_slice = torch.from_numpy(np.array(traj.observations)[start_idx : res_idx + 1, :])
        states_slice = (states_slice - self.state_mean) / self.state_std

        actions_slice = torch.from_numpy(np.array(traj.actions)[start_idx : res_idx + 1, :])
        rewards_slice = torch.from_numpy(np.array(traj.rewards)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)
        rtgs_slice = torch.from_numpy(np.array(traj.returns)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)

        # pad 0
        states_slice = torch.cat([torch.zeros(pad_len, states_slice.shape[-1]), states_slice], dim = 0)
        actions_slice = torch.cat([torch.zeros(pad_len, actions_slice.shape[-1]), actions_slice], dim = 0)
        rewards_slice = torch.cat([torch.zeros(pad_len, rewards_slice.shape[-1]), rtgs_slice], dim = 0)
        rtgs_slice = torch.cat([torch.zeros(pad_len, rtgs_slice.shape[-1]), rtgs_slice], dim = 0)

        if self.single_timestep: # take the last step
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[res_idx : res_idx + 1]) # (1,)
        else: 
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[start_idx : res_idx + 1]) #(real_ctx_len, )
            timesteps_slice = torch.cat([torch.zeros(pad_len), timesteps_slice], dim = 0)

        # print(f"Size of output: {states_slice.shape}, {actions_slice.shape}, {rtgs_slice.shape}, {timesteps_slice.shape}")
        # print(f"Dataset actions_slice: {actions_slice.shape}")
        if not self.keep_ctx:
            states_slice = states_slice[0,:]
            actions_slice = actions_slice[0,:]
            rewards_slice = rewards_slice[0,:]
            rtgs_slice = rtgs_slice[0,:]
            timesteps_slice = timesteps_slice[0]

        if self.with_mask:
            attn_mask = torch.cat([torch.zeros((pad_len)), torch.ones((ctx-pad_len))], dim=-1)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask, weight
        else:
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, weight
    
    def getitem(self, idx):
        if self.with_mask:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask, weight = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask, weight
        else:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, weight = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, weight           

    
    def get_max_return(self):
        traj_rets = [traj.returns[0] for traj in self._trajs]
        return max(traj_rets)
    
    def get_normalize_coef(self):
        '''
        Get state normalization mean and std
        '''
        return self.state_mean, self.state_std

class TrajCtxFloatLengthDataset(Dataset):
    '''
    Son of the pytorch Dataset class
    Provides context length, no next state.
    Trajectory length is uncertain
    '''

    def __init__(self, trajs, ctx = 1, single_timestep = False, keep_ctx = True, with_mask=False, state_normalize=False):    
        '''
        trajs: list(traj), namedtuple "observations", "actions", "rewards", "returns", "timesteps", "terminated", "truncated", "infos" \n
        single_timestep: bool. If true, timestep only keep initial step; Else (ctx,) \n
        keep_ctx: If False, ctx must be set 1, and we will not keep ctx dimension.
        with_mask: If true, also return attention mask. For DT
        state_normalize: If true, normalize states
        Note: Each traj must have same number of timesteps
        '''    
        # All 2d tensors n*args.horizon (timesteps n*(args.horizon+1))
        # self.obs = []
        # self.actions = []
        # self.rewards = []
        # self.rts = []
        # self.timesteps = []
        # for traj in trajs:
        #     self.obs += traj.observations #obss
        #     self.actions += traj.actions # np.array (num_trajectories, horizon)
        #     # self.done_idxs = done_idxs # What is this for?
        #     self.rewards += traj.rewards
        #     self.rtgs += traj.returns
        #     self.timesteps += traj.timesteps # (trajectory_num,horizon+1)
        #     # self._trajectory_num = states.shape[0] # number of trajectories in dataset
        #     # self._horizon = states.shape[1]
        self._trajs = trajs
        self._trajectory_num = len(self._trajs)
        self._horizon = len(self._trajs[0].observations)
        self.keep_ctx = keep_ctx
        self.with_mask = with_mask

        if not keep_ctx:
            assert ctx == 1, f"When keep_ctx = False, ctx must be 1"

        self.ctx = ctx
        self.single_timestep = single_timestep

        self.state_normalize = state_normalize

        if state_normalize:
            states_list = []
            for traj in trajs:
                states_list += traj.observations
            states = np.concatenate(states_list, axis = 0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        else:
            self.state_mean = 0
            self.state_std = 1

        self.traj_start_idxs = [] # The index of each traj's start
        cnt = 0
        self.traj_idx_list = [] # maintain the traj_idx of each idx
        for i,traj in enumerate(trajs):
            self.traj_start_idxs.append(cnt)
            traj_len = len(traj.rewards)
            self.traj_idx_list += [i for _ in range(traj_len)]
            cnt += traj_len
        self.traj_start_idxs.append(cnt) # Last idx is the total number of data


        # number of trajectories should match
        # assert self._trajectory_num == actions.shape[0] 
        # assert self._trajectory_num == rtgs.shape[0] 
        # assert self._trajectory_num == timesteps.shape[0] 
    
    def __len__(self):
        # return self._trajectory_num * self._horizon
        return self.traj_start_idxs[-1]
    
    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        '''
        Update: Also train incomplete contexts. Incomplete contexts pad 0.
        Input: idx, int, index to get an RTG trajectory slice from dataset \n
        Return: An RTG trajectory slice with length ctx_length \n
        - states: Tensor of size [ctx_length, state_space_size]
        - actions: Tensor of size [ctx_length, action_dim], here action is converted to one-hot representation
        - rewards: Tensor of size [ctx_length, 1]
        - rtgs: Tensor of size [ctx_length, 1]
        - timesteps: (ctx_length) if single_timestep=False; else (1,), only keep the first timestep
        Note: if keep_ctx = False, all returns above will remove the first dim. In particular, timesteps becomes scalar.
        '''
        # block_size = self.block_size // 3  # "//" means [5/3], here approx context length
        # done_idx = idx + block_size
        # for i in self.done_idxs:
        #     if i > idx: # first done_idx greater than idx
        #         done_idx = min(int(i), done_idx)
        #         break
        # idx = done_idx - block_size
        # states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        # states = states / 255.

        # print(f"Dataset: get index {idx}")

        ctx = self.ctx # context length
        # data_num_per_trajectory = self._horizon # number of data one trajectory provides
        # trajectory_idx = idx // data_num_per_trajectory # which trajectory to read, row
        # res_idx = idx - trajectory_idx * data_num_per_trajectory # column index to read

        # num_traj = len(self._trajs)
        # for i in range(self._trajectory_num -1,-1,-1):
        #     if self.traj_start_idxs[i] <= idx:
        #         break
        trajectory_idx = self.traj_idx_list[idx]
        res_idx = idx - self.traj_start_idxs[trajectory_idx]

        # print(f"trajectory_idx:{trajectory_idx}, res_idx: {res_idx}, start idx {self.traj_start_idxs[i]}")
        
        # assert res_idx < self._horizon, idx
        # assert trajectory_idx < self._trajectory_num, idx

        # Test whether it is full context length
        if res_idx - ctx + 1 < 0:
            start_idx = 0
            pad_len = ctx - res_idx - 1 # number of zeros to pad
        else:
            start_idx = res_idx - ctx + 1
            pad_len = 0

        traj = self._trajs[trajectory_idx]
        # print(f"Obs shape: {np.array(traj.observations).shape}")
        # states_slice = self.data[trajectory_idx, start_idx : res_idx + 1, :]
        # actions_slice = self.actions[trajectory_idx, start_idx : res_idx + 1, :]
        # rtgs_slice = self.rtgs[trajectory_idx, start_idx : res_idx + 1, :]
        states_slice = torch.from_numpy(np.array(traj.observations)[start_idx : res_idx + 1, :])
        states_slice = (states_slice - self.state_mean) / self.state_std

        # print(f"start: {start_idx}, end: {res_idx}")
        # print(f"Before cat: {states_slice.shape}")

        actions_slice = torch.from_numpy(np.array(traj.actions)[start_idx : res_idx + 1, :])
        rewards_slice = torch.from_numpy(np.array(traj.rewards)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)
        rtgs_slice = torch.from_numpy(np.array(traj.returns)[start_idx : res_idx + 1]).unsqueeze(-1) # (T,1)

        # pad 0
        states_slice = torch.cat([torch.zeros(pad_len, states_slice.shape[-1]), states_slice], dim = 0)
        actions_slice = torch.cat([torch.zeros(pad_len, actions_slice.shape[-1]), actions_slice], dim = 0)
        rewards_slice = torch.cat([torch.zeros(pad_len, rewards_slice.shape[-1]), rtgs_slice], dim = 0)
        rtgs_slice = torch.cat([torch.zeros(pad_len, rtgs_slice.shape[-1]), rtgs_slice], dim = 0)

        if self.single_timestep: # take the last step
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[res_idx : res_idx + 1]) # (1,)
        else: 
            timesteps_slice = torch.from_numpy(np.array(traj.timesteps)[start_idx : res_idx + 1]) #(real_ctx_len, )
            timesteps_slice = torch.cat([torch.zeros(pad_len), timesteps_slice], dim = 0)

        # print(f"Size of output: {states_slice.shape}, {actions_slice.shape}, {rtgs_slice.shape}, {timesteps_slice.shape}")
        # print(f"Dataset actions_slice: {actions_slice.shape}")
        if not self.keep_ctx:
            states_slice = states_slice[0,:]
            actions_slice = actions_slice[0,:]
            rewards_slice = rewards_slice[0,:]
            rtgs_slice = rtgs_slice[0,:]
            timesteps_slice = timesteps_slice[0]

        assert states_slice.shape[0] != 0, f"{idx}, {states_slice.shape}"
        if self.with_mask:
            attn_mask = torch.cat([torch.zeros((pad_len)), torch.ones((ctx-pad_len))], dim=-1)
            # print(f"Dataset: Successfully get index {idx}")
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask
        else:
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice
    
    def getitem(self, idx):
        if self.with_mask:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice, attn_mask
        else:
            states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice = self.__getitem__(idx)
            return states_slice, actions_slice, rewards_slice, rtgs_slice, timesteps_slice           

    
    def get_max_return(self):
        traj_rets = [traj.returns[0] for traj in self._trajs]
        return max(traj_rets)
    
    def get_normalize_coef(self):
        '''
        Get state normalization mean and std
        '''
        return self.state_mean, self.state_std