'''
Learn the MDP model, including transition and dynamics
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
# from torch.distributions import Normal
# from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.transforms import TanhTransform
from .fc_network import FullyConnectedNetwork

class DynamicsModel(nn.Module):
    '''
    Include transition and dynamics prediction.
    '''
    def __init__(self, state_dim, action_dim, arch = '256-256', independent_network = False):
        '''
        independent_network: If True, use two separate networks for state, reward prediction
        '''
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = arch
        self.independent_network = independent_network

        if independent_network:
            self.s_network = FullyConnectedNetwork(input_dim = state_dim + action_dim + 1, 
                                                   output_dim= state_dim,
                                                   arch = arch)
            self.r_network = FullyConnectedNetwork(input_dim = state_dim + action_dim + 1, 
                                                   output_dim= 1,
                                                   arch = arch)     
        else:       
            self.network = FullyConnectedNetwork(input_dim = state_dim + action_dim + 1, 
                                             output_dim= 1 + state_dim,
                                             arch = arch)

    def forward(self, state, action, timestep=None):
        '''
        Input
        ===
        - state, (batch, state_dim) / (state_dim).
        - action, (batch, action_dim) / (action_dim)
        - timestep, (batch, ) / Scalar

        Output
        ===
        - pred_reward, (batch,) or scalar
        - pred_next_state, (batch, state_dim) / (state_dim)
        '''
        assert state.shape[-1] == self.state_dim, f"State shape is expected to be {self.state_dim}, got {state.shape[-1]}!"
        assert action.shape[-1] == self.action_dim, f"State shape is expected to be {self.action_dim}, got {action.shape[-1]}!"

        if timestep is not None:
            timestep = timestep.unsqueeze(-1).type(torch.float32) # (batch, 1) / (1,)
        else:
            if state.dim() == 2: # has batch
                timestep = torch.zeros((state.shape[0], 1)).to(state.device) # Pad with 0 (batch, 1)
            else: # no batch
                timestep = torch.zeros((1)).to(state.device) # Pad with 0 (1)

        input = torch.cat([state, action, timestep], dim=-1)

        if self.independent_network:
            pred_reward = self.r_network(input).squeeze(dim=-1) # (batch,) or scalar
            pred_next_state = self.s_network(input)
        else:
            output = self.network(input)

            assert output.dim() == 1 or output.dim() == 2, f"Output dim is expected to be 1 or 2, got {output.dim()}!"

            if output.dim() == 1: # no batch
                pred_reward = output[0]
                pred_next_state = output[1:]
            else: # Get batch
                pred_reward = output[:,0]
                pred_next_state = output[:,1:]

        return pred_reward, pred_next_state
    
class NextStateModel(nn.Module):
    '''
    Include transition and dynamics prediction.
    '''
    def __init__(self, state_dim, action_dim, arch = '256-256'):
        '''
        independent_network: If True, use two separate networks for state, reward prediction
        '''
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = arch
      
        self.network = FullyConnectedNetwork(input_dim = state_dim + action_dim + 1, 
                                            output_dim= state_dim,
                                            arch = arch)

    def forward(self, state, action, timestep=None):
        '''
        Input
        ===
        - state, (batch, state_dim) / (state_dim).
        - action, (batch, action_dim) / (action_dim)
        - timestep, (batch, ) / Scalar

        Output
        ===
        - pred_reward, (batch,) or scalar
        - pred_next_state, (batch, state_dim) / (state_dim)
        '''
        assert state.shape[-1] == self.state_dim, f"State shape is expected to be {self.state_dim}, got {state.shape[-1]}!"
        assert action.shape[-1] == self.action_dim, f"State shape is expected to be {self.action_dim}, got {action.shape[-1]}!"

        if timestep is not None:
            timestep = timestep.unsqueeze(-1).type(torch.float32) # (batch, 1) / (1,)
        else:
            if state.dim() == 2: # has batch
                timestep = torch.zeros((state.shape[0], 1)).to(state.device) # Pad with 0 (batch, 1)
            else: # no batch
                timestep = torch.zeros((1)).to(state.device) # Pad with 0 (1)

        input = torch.cat([state, action, timestep], dim=-1)

        pred_next_state = self.network(input)

        return pred_next_state
    
class RewardModel(nn.Module):
    '''
    Include transition and dynamics prediction.
    '''
    def __init__(self, state_dim, action_dim, arch = '256-256'):
        '''
        independent_network: If True, use two separate networks for state, reward prediction
        '''
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = arch
          
        self.network = FullyConnectedNetwork(input_dim = state_dim + action_dim + 1, 
                                            output_dim= 1,
                                            arch = arch)

    def forward(self, state, action, timestep=None):
        '''
        Input
        ===
        - state, (batch, state_dim) / (state_dim).
        - action, (batch, action_dim) / (action_dim)
        - timestep, (batch, ) / Scalar

        Output
        ===
        - pred_reward, (batch,) or scalar
        - pred_next_state, (batch, state_dim) / (state_dim)
        '''
        assert state.shape[-1] == self.state_dim, f"State shape is expected to be {self.state_dim}, got {state.shape[-1]}!"
        assert action.shape[-1] == self.action_dim, f"State shape is expected to be {self.action_dim}, got {action.shape[-1]}!"

        if timestep is not None:
            timestep = timestep.unsqueeze(-1).type(torch.float32) # (batch, 1) / (1,)
        else:
            if state.dim() == 2: # has batch
                timestep = torch.zeros((state.shape[0], 1)).to(state.device) # Pad with 0 (batch, 1)
            else: # no batch
                timestep = torch.zeros((1)).to(state.device) # Pad with 0 (1)

        input = torch.cat([state, action, timestep], dim=-1)

        pred_reward = self.network(input).squeeze(dim=-1) # (batch,) or scalar

        return pred_reward


class InitStateModel(nn.Module):
    '''
    Learn the initial state distribution from dataset
    '''
    def __init__(self, state_dim, n_support):
        super().__init__()
        self.state_dim = state_dim
        self.n_support = n_support
        self.state_network = nn.Embedding(1, self.state_dim * self.n_support)
        self.prob_network = nn.Embedding(1, self.n_support)
    def forward(self, dummy=None):
        '''
        dummy: any Tensor, used to reveal the tensor device. If None, default to be cpu
        Return:
        - support_states: (n_support, state_dim)
        - support_probs: (n_support)
        '''
        dummy_input = torch.tensor(0).long()
        if dummy is not None:
            dummy_input = dummy_input.to(dummy.device)
        support_states = self.state_network(dummy_input).reshape(self.n_support, self.state_dim)
        support_probs = self.prob_network(dummy_input)
        support_probs = torch.softmax(support_probs, dim=-1)
        return support_states, support_probs

class InverseDynamics(nn.Module):
    '''
    Compute action to reach desired state
    '''
    def __init__(self, state_dim, action_dim, arch = '256-256'):
        '''
        independent_network: If True, use two separate networks for state, reward prediction
        '''
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = arch
   
        self.network = FullyConnectedNetwork(input_dim = 2*state_dim + 1, 
                                            output_dim= action_dim,
                                            arch = arch)

    def forward(self, state, next_state, timestep=None):
        '''
        Input
        ===
        - state, (batch, state_dim) / (state_dim).
        - next_state, (batch, state_dim) / (state_dim)
        - timestep, (batch, ) / Scalar / None

        Output
        ===
        - pred_action, (batch, action_dim) / (action_dim)
        '''
        # assert state.shape[-1] == self.state_dim, f"State shape is expected to be {self.state_dim}, got {state.shape[-1]}!"
        # assert action.shape[-1] == self.action_dim, f"State shape is expected to be {self.action_dim}, got {action.shape[-1]}!"
        
        if timestep is not None:
            timestep = timestep.unsqueeze(-1).type(torch.float32) # (batch, 1) / (1,)
        else:
            if state.dim() == 2: # has batch
                timestep = torch.zeros((state.shape[0], 1)).to(state.device) # Pad with 0 (batch, 1)
            else: # no batch
                timestep = torch.zeros((1)).to(state.device) # Pad with 0 (1)
        input = torch.cat([state, next_state, timestep], dim=-1)

        pred_action = self.network(input)

        return pred_action


        



