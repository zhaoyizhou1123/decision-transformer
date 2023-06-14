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
        if arch == '' or arch == '/': # No hidden layers
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

    def __init__(self, observation_dim, action_dim, horizon, arch='256-256', 
                 token_repeat=1, orthogonal_init=False, embd_dim = -1):
        '''
        - embd_dim: int. If <= 0, no embedding, simply input (s,a,g,t); Else embed(s)+embed(t), ...
        - horizon: used for timestep embedding. Timestep starts with 0
        - token_repeat: int, repeat tokens multiple times to emphasize it.
        '''
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.embd_dim = embd_dim
        self.arch = arch
        self.orthogonal_init = orthogonal_init
        self.token_repeat = int(token_repeat)

        self.do_embd = (embd_dim > 0) # If True, do embedding, else simply (s,g,a,t)

        if self.do_embd:
            self.embd_obs = nn.Linear(observation_dim, embd_dim)
            self.embd_action = nn.Linear(action_dim, embd_dim)
            self.embd_timestep = nn.Embedding(horizon, embd_dim)
            self.network = FullyConnectedNetwork(
                2*embd_dim*token_repeat, 1, arch, orthogonal_init
            )
        else:
            self.network = FullyConnectedNetwork(
                token_repeat * (observation_dim + action_dim  + 1), 1, arch, orthogonal_init
            )
    

    def forward(self, observations, actions, timesteps):
        '''
        - observations: (batch, obs_dim) or (obs_dim)
        - actions: (batch, action_dim) or (action_dim)
        - timesteps: (batch) or scalar
        Return: Tensor (batch,) or scalar
        '''
        actions = actions.type(torch.float)
        if not isinstance(timesteps, torch.Tensor): # scalar
            timesteps = torch.tensor(timesteps) # (1)

        if self.do_embd:
            timesteps = timesteps.long()
            obs_embd = self.embd_obs(observations) + self.embd_timestep(timesteps)
            # print(f"model_cql: action shape {actions.shape}")
            action_embd = self.embd_action(actions) + self.embd_timestep(timesteps)
            input_tensor = torch.cat([obs_embd, action_embd], dim=-1)
        else:
            # An easer encoding (obs,action,timestep)
            timesteps = timesteps.unsqueeze(-1) # (batch, 1) or (1,1)
            # assert observations.dim() == actions.dim() and actions.dim() == timesteps.dim(), f"Dim mismatch: {observations.shape}, {actions.shape}, {timesteps.shape}"
            # repeat_actions = [actions for i in range(self.token_repeat)] # repeat actions
            input_list = [observations] + [actions] + [timesteps]
            input_tensor = torch.cat(input_list, dim=-1)
        input_tensor = input_tensor.repeat(1,self.token_repeat) # repeat 
        return torch.squeeze(self.network(input_tensor), dim=-1)
    
# model = FullyConnectedQFunction(1,2,4,20,'').network.network
# print(model)
# print(model[0].weight.data)
# print(model[0].bias.data)