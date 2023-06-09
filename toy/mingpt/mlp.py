# Modfied from https://github.com/young-geng/CQL/blob/master/SimpleSAC/model.py#L42
# MLP model for policy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
# from torch.distributions import Normal
# from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.transforms import TanhTransform

class FullyConnectedNetwork(nn.Module):
    '''
    Sequential fully-connected layers, with ReLU
    '''
    def __init__(self, input_dim, output_dim, arch='256-256'):
        '''
        arch: specifies dim of hidden layers, separated by '-'. We only want one layer, so 'int'
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch

        d = input_dim
        modules = []
        if arch == '': # No hidden layers
            hidden_sizes = []
        else:
            hidden_sizes = [int(h) for h in arch.split('-')]

        for hidden_size in hidden_sizes:
            fc = nn.Linear(d, hidden_size)
            modules.append(fc)
            modules.append(nn.ReLU())
            d = hidden_size

        last_fc = nn.Linear(d, output_dim)
        nn.init.xavier_uniform_(last_fc.weight, gain=1e-2)

        nn.init.constant_(last_fc.bias, 0.0)
        modules.append(last_fc)

        self.network = nn.Sequential(*modules)

    def forward(self, input_tensor):
        return self.network(input_tensor)
    
class MlpPolicy(nn.Module):
    '''
    Model for policy
    '''

    def __init__(self, observation_dim, n_act, ctx, token_repeat=1, arch='256-256'):
        '''
        - n_act: We expect acts given in one-hot, so action_dim = n_action
        - embd_dim: dimension of observation/action embedding
        - ctx: context length
        - horizon: used for timestep embedding. Timestep starts with 0
        - action_repeat: int, repeat action multiple times to emphasize it.
        '''
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = n_act
        self.arch = arch
        self.ctx = ctx
        self.token_repeat = int(token_repeat)

        # s,g,t * ctx, a * ctx-1
        self.network = FullyConnectedNetwork(
            self.token_repeat*(ctx * (observation_dim + 1 + 1) + (ctx-1) * self.action_dim), 
            self.action_dim, arch
        )

    def forward(self, obs, acts, rtgs, timesteps):
        '''
        - obs: (batch, T, obs_dim). T <= ctx
        - acts: (batch, T-1, 1). The last action is excluded.
        - rtgs: (batch, T, 1)
        - timesteps: (batch, T)
        Return: Tensor (batch, action_dim), representing stage policy after softmax
        '''
        batch = obs.shape[0]

        # Convert actions to one-hot
        assert acts.shape[-1] == 1, f"Invalid action_dim {acts.shape[-1]}"
        acts = acts.squeeze(dim=-1) # (batch,ctx)
        acts = f.one_hot(acts.long(),self.action_dim) # (batch, ctx, n_act)

        obs = obs.type(torch.float32)
        acts = acts.type(torch.float32)
        rtgs = rtgs.type(torch.float32)
        timesteps = timesteps.unsqueeze(-1).type(torch.float32) # (batch, T, 1)

        # Pad to ctx length. Timesteps pad with -1, acts pad to ctx-1
        obs = torch.cat([torch.zeros(batch, self.ctx-obs.shape[1], obs.shape[-1]).to(obs.device), obs], dim=1)
        acts = torch.cat([torch.zeros(batch, self.ctx-1-acts.shape[1], acts.shape[-1]).to(acts.device), acts], dim=1)
        rtgs = torch.cat([torch.zeros(batch, self.ctx-rtgs.shape[1], rtgs.shape[-1]).to(rtgs.device), rtgs], dim=1)
        timesteps = torch.cat([torch.zeros(batch, self.ctx-timesteps.shape[1], timesteps.shape[-1]).to(timesteps.device)-1, timesteps], dim=1)

        # Reshape
        obs = obs.reshape(batch, -1) # (batch, ctx*obs_dim)
        acts = acts.reshape(batch, -1) # (batch, (ctx-1)*act_dim)
        rtgs = rtgs.reshape(batch, -1) # (batch, ctx*1)
        timesteps = timesteps.reshape(batch, -1) # (batch, ctx*1)


        # Order of tokens is unimportant for mlp, so we don't use interleave
        input_tensor = torch.cat([timesteps, obs, rtgs, acts], dim=-1)  
        input_tensor = input_tensor.repeat(1,self.token_repeat) # repeat tokens

        # (batch, action_dim)      
        return self.network(input_tensor)
    
# model = FullyConnectedQFunction(1,2,4,20,'').network.network
# print(model)
# print(model[0].weight.data)
# print(model[0].bias.data)