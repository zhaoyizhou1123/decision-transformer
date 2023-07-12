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
    def __init__(self, state_dim, action_dim, arch = '256-256'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.arch = arch
        self.network = FullyConnectedNetwork(input_dim = state_dim + action_dim + 1, 
                                             output_dim= 1 + state_dim,
                                             arch = arch)

    def forward(self, state, action, timestep):
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

        input = torch.cat([state, action, timestep.unsqueeze(-1)], dim=-1)
        output = self.network(input)

        assert output.dim() == 1 or output.dim() == 2, f"Output dim is expected to be 1 or 2, got {output.dim()}!"

        if output.dim() == 1: # no batch
            pred_reward = output[0]
            pred_next_state = output[1:]
        else: # Get batch
            pred_reward = output[:,0]
            pred_next_state = output[:,1:]

        return pred_reward, pred_next_state

        



