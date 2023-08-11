# Modfied from https://github.com/young-geng/CQL/blob/master/SimpleSAC/model.py#L42
# MLP model for policy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
# from torch.distributions import Normal
# from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.transforms import TanhTransform
from .fc_network import FullyConnectedNetwork
    
class RcslPolicy(nn.Module):
    '''
    Model for output policy
    '''

    def __init__(self, observation_dim, action_dim, ctx, horizon, token_repeat=1, arch='256-256', embd_dim = -1, simple_input=False):
        '''
        - action_dim
        - ctx: context length
        - horizon: used for timestep embedding. Timestep starts with 0
        - token_repeat: int, repeat action multiple times to emphasize it.
        - embd_dim: int. If <= 0, no embedding, simply input (s,a,g,t); Else embed(s)+embed(t), ...
        - simple_input: If true, only use input (g_{t-ctx+1},g_{t-ctx+2},...g_{t-1},s_t,g_t,t) as input
        '''
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        self.ctx = ctx
        self.token_repeat = int(token_repeat)
        self.embd_dim = embd_dim

        self.do_embd = (embd_dim > 0) # If True, do embedding, else simply (s,g,a,t)
        self.simple_input = simple_input

        if self.do_embd:
            self.embd_obs = nn.Linear(observation_dim, embd_dim)
            self.embd_action = nn.Linear(self.action_dim, embd_dim)
            self.embd_rtg = nn.Linear(1, embd_dim)
            self.embd_timestep = nn.Embedding(horizon, embd_dim)

        # s,g,t * ctx, a * ctx-1
        if self.do_embd:
            if self.simple_input:
                input_dim = (ctx + 1) * embd_dim
            else:
                input_dim = ctx * embd_dim * 2 + (ctx-1) * embd_dim
        else:
            if self.simple_input:
                input_dim = self.observation_dim + ctx + 1
            else:
                input_dim = ctx * (self.observation_dim + 1 + 1) + (ctx-1) * self.action_dim
        self.network = FullyConnectedNetwork(
            self.token_repeat * input_dim, 
            action_dim, arch
        )

    def forward(self, obs, acts, rtgs, timesteps):
        '''
        - obs: (batch, T, obs_dim). T <= ctx
        - acts: (batch, T-1, 1). The last action is excluded.
        - rtgs: (batch, T, 1)
        - timesteps: (batch, T)
        Return: Tensor (batch, action_dim), representing the chosen action
        '''
        batch = obs.shape[0]

        # Convert actions to one-hot
        # assert acts.shape[-1] == 1, f"Invalid action_dim {acts.shape[-1]}"

        # if self.one_hot:
        #     acts = acts.squeeze(dim=-1) # (batch,T-1)
        #     acts = f.one_hot(acts.long(),self.action_dim) # (batch, T-1, action_dim)

        obs = obs.type(torch.float32)
        acts = acts.type(torch.float32)
        rtgs = rtgs.type(torch.float32)

        if self.do_embd: # embed obs, acts and rtgs
            timesteps = timesteps.long()
            obs = self.embd_obs(obs) + self.embd_timestep(timesteps)
            acts = self.embd_action(acts) + self.embd_timestep(timesteps[:, :-1]) # exclude the last timestep
            rtgs = self.embd_rtg(rtgs) + self.embd_timestep(timesteps)
        else:
            timesteps = timesteps.unsqueeze(-1).type(torch.float32) # (batch, T, 1)

        # Pad to ctx length. Timesteps pad with -1, acts pad to ctx-1
        obs = torch.cat([torch.zeros(batch, self.ctx-obs.shape[1], obs.shape[-1]).to(obs.device), obs], dim=1) # (batch, ctx, obs_dim)
        acts = torch.cat([torch.zeros(batch, self.ctx-1-acts.shape[1], acts.shape[-1]).to(acts.device), acts], dim=1)
        rtgs = torch.cat([torch.zeros(batch, self.ctx-rtgs.shape[1], rtgs.shape[-1]).to(rtgs.device), rtgs], dim=1)

        if not self.do_embd:
            timesteps = torch.cat([torch.zeros(batch, self.ctx-timesteps.shape[1], timesteps.shape[-1]).to(timesteps.device)-1, timesteps], dim=1)

        # Reshape
        if self.simple_input:
            obs = obs[:, -1, :] # (batch, obs_dim) or (batch, embd_dim)
        else:
            obs = obs.reshape(batch, -1) # (batch, ctx*obs_dim) or (batch, ctx*embd_dim)
        
        # Don't use action in simple_input, so only consider no simple input
        acts = acts.reshape(batch, -1) # (batch, (ctx-1)*act_dim) or (batch, (ctx-1)*embd_dim)

        # Rtgs always keep full length
        rtgs = rtgs.reshape(batch, -1) # (batch, ctx*1) or (batch, ctx*embd_dim)

        if not self.do_embd:
            timesteps = timesteps.reshape(batch, -1) # (batch, ctx*1)
            if self.simple_input:
                timesteps = timesteps[:, -1:] # (batch, 1)


        # Order of tokens is unimportant for mlp, so we don't use interleave
        if self.simple_input:
            if not self.do_embd:
                input_tensor = torch.cat([timesteps, obs, rtgs], dim=-1) # (batch, 1) (batch, 1*obs_dim), (batch, ctx)
            else:
                input_tensor = torch.cat([obs, rtgs], dim=-1) # (batch, 1*embd_dim), (batch, ctx)
        else:
            if not self.do_embd:
                input_tensor = torch.cat([timesteps, obs, rtgs, acts], dim=-1)  
            else:
                input_tensor = torch.cat([obs, rtgs, acts], dim=-1)
            input_tensor = input_tensor.repeat(1,self.token_repeat) # repeat tokens

        # (batch, action_dim)      
        return self.network(input_tensor)
    
class BehaviorPolicy(nn.Module):
    '''
    Model for dataset behavior policy, simply pi(a|s,t)
    '''

    def __init__(self, observation_dim, action_dim, token_repeat=1, arch='256-256'):
        '''
        - action_dim
        - ctx: context length
        - horizon: used for timestep embedding. Timestep starts with 0
        - token_repeat: int, repeat action multiple times to emphasize it.
        '''
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        # self.ctx = ctx
        self.token_repeat = int(token_repeat)

        input_dim = self.observation_dim + 1 # (s,t)
        self.network = FullyConnectedNetwork(
            self.token_repeat * input_dim, 
            action_dim, arch
        )

    def forward(self, obs, timesteps):
        '''
        - obs: (batch, obs_dim). T <= ctx
        - timesteps: (batch,)
        Return: Tensor (batch, action_dim), representing the chosen action
        '''
        batch = obs.shape[0]

        # Convert actions to one-hot
        # assert acts.shape[-1] == 1, f"Invalid action_dim {acts.shape[-1]}"

        # if self.one_hot:
        #     acts = acts.squeeze(dim=-1) # (batch,T-1)
        #     acts = f.one_hot(acts.long(),self.action_dim) # (batch, T-1, action_dim)

        obs = obs.type(torch.float32)
        timesteps = timesteps.unsqueeze(-1).type(torch.float32) # (batch, 1)

        input_tensor = torch.cat([obs, timesteps], dim=-1) # (batch, obs_dim + 1)
        input_tensor = input_tensor.repeat(1,self.token_repeat) # repeat tokens

        # (batch, action_dim)      
        return self.network(input_tensor)
    
class StochasticPolicy(nn.Module):
    '''
    Model for dataset behavior policy, simply pi(a|s,t), but can output stochastic policy
    Update: Remove prob output. Use uniform sample
    '''

    def __init__(self, observation_dim, action_dim, token_repeat=1, arch='256-256', n_support = 2):
        '''
        - action_dim
        - ctx: context length
        - horizon: used for timestep embedding. Timestep starts with 0
        - token_repeat: int, repeat action multiple times to emphasize it.
        - n_support: int, number of support action at output
        '''
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.arch = arch
        # self.ctx = ctx
        self.token_repeat = int(token_repeat)
        self.n_support = n_support

        input_dim = self.observation_dim + 1 # (s,t)
        self.network = FullyConnectedNetwork(
            self.token_repeat * input_dim, 
            action_dim  * self.n_support, arch
        )

        # Bound the output logit
        # self.logit_abs_bound = 100

    def forward(self, obs, timesteps=None):
        '''
        - obs: (batch, obs_dim). T <= ctx
        - timesteps: (batch,) or None. If none, no timesteps input
        Return: support_actions (batch, n_support, action_dim)
        '''
        batch = obs.shape[0]

        # Convert actions to one-hot
        # assert acts.shape[-1] == 1, f"Invalid action_dim {acts.shape[-1]}"

        # if self.one_hot:
        #     acts = acts.squeeze(dim=-1) # (batch,T-1)
        #     acts = f.one_hot(acts.long(),self.action_dim) # (batch, T-1, action_dim)

        obs = obs.type(torch.float32)

        if timesteps is not None:
            timesteps = timesteps.unsqueeze(-1).type(torch.float32) # (batch, 1)
        else:
            timesteps = torch.zeros((batch, 1)).to(obs.device) # Pad with 0 (batch, 1)

        input_tensor = torch.cat([obs, timesteps], dim=-1) # (batch, obs_dim + 1)
        input_tensor = input_tensor.repeat(1,self.token_repeat) # repeat tokens

        output_tensor = self.network(input_tensor) #(batch, action_dim*n_support)
        # support_actions = output_tensor[:,0:self.action_dim * self.n_support].reshape(-1, self.n_support, self.action_dim)
        support_actions = output_tensor.reshape(-1, self.n_support, self.action_dim)
        # support_probs = output_tensor[:, self.action_dim * self.n_support : ] # (batch, n_support) 
        # clamp_support_probs = torch.clamp(support_probs, min=-self.logit_abs_bound, max=self.logit_abs_bound)

        # check if there is nan in output
        # assert np.isnan(torch.softmax(clamp_support_probs, dim=-1).detach().cpu().numpy()).any() == False, f"Nan occurs for logits {support_probs}"

        # Truncate action to [-1,1] for each coordinate by pointmaze. Perform softmax for output probs
        # return torch.clamp(support_actions, min=-1, max=1), torch.softmax(clamp_support_probs, dim=-1)
        return support_actions

class StochasticState(nn.Module):
    '''
    Model for dataset behavior policy, directly choose the next state
    '''

    def __init__(self, observation_dim, token_repeat=1, arch='256-256', n_support = 2):
        '''
        - action_dim
        - ctx: context length
        - horizon: used for timestep embedding. Timestep starts with 0
        - token_repeat: int, repeat action multiple times to emphasize it.
        - n_support: int, number of support action at output
        '''
        super().__init__()
        self.observation_dim = observation_dim
        # self.action_dim = action_dim
        self.arch = arch
        # self.ctx = ctx
        self.token_repeat = int(token_repeat)
        self.n_support = n_support

        input_dim = self.observation_dim + 1 # (s,t)
        self.network = FullyConnectedNetwork(
            self.token_repeat * input_dim, 
            (self.observation_dim + 1) * self.n_support, arch
        )

        # Bound the output logit
        self.logit_abs_bound = 100

    def forward(self, obs, timesteps=None):
        '''
        - obs: (batch, obs_dim). T <= ctx
        - timesteps: (batch,) or None. If none, no timesteps input
        Return: support_actions (batch, n_support, action_dim), probs (batch, n_support)
        '''
        batch = obs.shape[0]

        # Convert actions to one-hot
        # assert acts.shape[-1] == 1, f"Invalid action_dim {acts.shape[-1]}"

        # if self.one_hot:
        #     acts = acts.squeeze(dim=-1) # (batch,T-1)
        #     acts = f.one_hot(acts.long(),self.action_dim) # (batch, T-1, action_dim)

        obs = obs.type(torch.float32)

        if timesteps is not None:
            timesteps = timesteps.unsqueeze(-1).type(torch.float32) # (batch, 1)
        else:
            timesteps = torch.zeros((batch, 1)).to(obs.device) # Pad with 0 (batch, 1)

        input_tensor = torch.cat([obs, timesteps], dim=-1) # (batch, obs_dim + 1)
        input_tensor = input_tensor.repeat(1,self.token_repeat) # repeat tokens

        output_tensor = self.network(input_tensor) #(batch, (action_dim+1)*n_support)
        support_states = output_tensor[:,0:self.observation_dim * self.n_support].reshape(-1, self.n_support, self.observation_dim)
        support_probs = output_tensor[:, self.observation_dim * self.n_support : ] # (batch, n_support) 
        clamp_support_probs = torch.clamp(support_probs, min=-self.logit_abs_bound, max=self.logit_abs_bound)

        # check if there is nan in output
        assert np.isnan(torch.softmax(clamp_support_probs, dim=-1).detach().cpu().numpy()).any() == False, f"Nan occurs for logits {support_probs}"

        # Perform softmax for output probs
        return support_states, torch.softmax(clamp_support_probs, dim=-1)

# model = FullyConnectedQFunction(1,2,4,20,'').network.network
# print(model)
# print(model[0].weight.data)
# print(model[0].bias.data)