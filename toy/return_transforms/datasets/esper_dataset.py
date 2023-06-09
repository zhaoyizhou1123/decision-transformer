from torch.utils.data import IterableDataset
import torch
import numpy as np
from return_transforms.utils.utils import return_labels

class ESPERDataset(IterableDataset):

    rand: np.random.Generator

    def __init__(self, trajs, n_actions, horizon, gamma=1, act_type='discrete',
                 epoch_len=1e5):
        '''
        trajs: list of Trajectory object
        horizon: true horizon + 1
        '''
        self.trajs = trajs
        self.rets = [return_labels(traj, gamma)
                     for traj in self.trajs] # list[list] num_trajs * horizon
        self.n_actions = n_actions
        self.horizon = horizon
        self.epoch_len = epoch_len
        self.act_type = act_type

    def segment_generator(self, epoch_len):
        for _ in range(epoch_len):
            traj_idx = self.rand.integers(len(self.trajs))
            traj = self.trajs[traj_idx]
            rets = self.rets[traj_idx] # list
            # if self.act_type == 'discrete':
            #     a = np.array(traj.actions)
            #     actions = np.zeros((a.size, self.n_actions))
            #     actions[np.arange(a.size), a] = 1
            # else:
            actions = np.array(traj.actions) # (ctx, action_dim)
            obs = np.array(traj.obs) # (ctx, action_dim)
            timesteps = np.array(traj.timesteps) # (ctx,)

            # assert timesteps.shape[0] == self.horizon-1, f"Mismatch timesteps length {timesteps.shape[0]}, horizon {self.horizon}"

            padded_obs = np.zeros((self.horizon, *obs.shape[1:])) # *tuple unpacks the tuple
            padded_acts = np.zeros((self.horizon, self.n_actions))
            padded_rets = np.zeros(self.horizon)
            padded_timesteps = np.zeros(self.horizon)

            padded_obs[-obs.shape[0]:] = obs

            assert actions.shape[-1] == self.n_actions, f"Mismatch action dim {actions.shape[-1]}"
            padded_acts[-obs.shape[0]:] = actions
            padded_rets[-obs.shape[0]:] = np.array(rets)
            padded_timesteps[-obs.shape[0]:] = timesteps
            seq_length = obs.shape[0] # horizon

            yield torch.tensor(padded_obs).float(), \
                torch.tensor(padded_acts).float(), \
                torch.tensor(padded_rets).float(), \
                torch.tensor(padded_timesteps).float(), \
                torch.tensor(seq_length).long()

    def __len__(self):
        return int(self.epoch_len)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self.rand = np.random.default_rng(None)
        if worker_info is None:  # single-process data loading, return the full iterator
            gen = self.segment_generator(int(self.epoch_len))
        else:  # in a worker process
            # split workload
            per_worker_time_steps = int(
                self.epoch_len / float(worker_info.num_workers))
            gen = self.segment_generator(per_worker_time_steps)
        return gen
