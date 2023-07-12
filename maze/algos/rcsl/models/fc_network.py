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
        arch: specifies dim of hidden layers, separated by '-'. We only want one layer, so 'int'. \n
        Upd: arch = '/' is equivalent to '', so it can be run in cmd
        '''
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.arch = arch

        d = input_dim
        modules = []
        if arch == '' or arch == '/': # No hidden layers
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