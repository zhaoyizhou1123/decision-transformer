# Modfied from https://github.com/young-geng/CQL/blob/master/SimpleSAC/model.py#L42
# Model for Q-function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.distributions import Normal
# from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.transforms import TanhTransform

class FullyConnectedNetwork(nn.Module):
    '''
    Sequential fully-connected layers, with ReLU
    '''
    def __init__(self, input_dim, output_dim, arch='256-256', orthogonal_init=False):
        '''
        arch: specifies dim of hidden layers, separated by '-'. We only want one layer, so 'int'
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init

        d = input_dim
        modules = []
        if arch == '': # No hidden layers
            hidden_sizes = []
        else:
            hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            if orthogonal_init:
                nn.init.orthogonal_(fc.weight, gain=np.sqrt(2))
                nn.init.constant_(fc.bias, 0.0)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        if orthogonal_init:
            nn.init.orthogonal_(last_fc.weight, gain=1e-2)
        else:
            nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)
    
class FullyConnectedQFunction(nn.Module):
    '''
    Model for Q-function
    '''

    def __init__(self, observation_dim, action_dim, embd_dim, horizon, arch='256-256', 
                 action_repeat=1, orthogonal_init=False):
        '''
        - embd_dim: dimension of observation/action embedding
        - horizon: used for timestep embedding. Timestep starts with 0
        - action_repeat: int, repeat action multiple times to emphasize it.
        '''
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.embd_dim = embd_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        # self.network = FullyConnectedNetwork(
        #     2*embd_dim, 1, arch, orthogonal_init
        # )
        self.network = FullyConnectedNetwork(
            observation_dim + action_dim * action_repeat + 1, 1, arch, orthogonal_init
        )
        self.action_repeat = action_repeat
        # self.embd_obs = nn.Linear(observation_dim, embd_dim)
        # self.embd_action = nn.Linear(action_dim, embd_dim)
        # self.embd_timestep = nn.Embedding(horizon, embd_dim)

    def forward(self, observations, actions, timesteps):
        '''
        - observations: (batch, obs_dim) or (obs_dim)
        - actions: (batch, action_dim) or (action_dim)
        - timesteps: (batch) or scalar
        Return: Tensor (batch,) or scalar
        '''
        actions = actions.type(torch.float)
        # obs_embd = self.embd_obs(observations) + self.embd_timestep(timesteps)
        # action_embd = self.embd_action(actions) + self.embd_timestep(timesteps)
        # input_tensor = torch.cat([obs_embd, action_embd], dim=-1)

        # An easer encoding (obs,action,timestep)
        if not isinstance(timesteps, torch.Tensor): # scalar
            timesteps = torch.tensor(timesteps) # (1)
        timesteps = timesteps.unsqueeze(-1) # (batch, 1) or (1,1)
        # assert observations.dim() == actions.dim() and actions.dim() == timesteps.dim(), f"Dim mismatch: {observations.shape}, {actions.shape}, {timesteps.shape}"
        repeat_actions = [actions for i in range(self.action_repeat)] # repeat actions
        input_list = [observations] + repeat_actions + [timesteps]
        input_tensor = torch.cat(input_list, dim=-1)
        return torch.squeeze(self.network(input_tensor), dim=-1)
    
# model = FullyConnectedQFunction(1,2,4,20,'').network.network
# print(model)
# print(model[0].weight.data)
# print(model[0].bias.data)