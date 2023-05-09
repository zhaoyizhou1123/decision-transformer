# Bugs: positional embedding double-check

"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

'''
Main modifications:
Single attention, without drop-out and layer normalization
'''

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np

class DTConfig:
    """ DT configuration """

    def __init__(self, state_dim, n_act, n_embd, horizon, ctx,                  
                 init_att = None, freeze_att = False,**kwargs):
        '''
        ctx: context length
        '''
        self.state_dim = state_dim
        self.n_act = n_act
        self.n_embd = n_embd
        self.horizon = horizon
        self.seq_length = 3*ctx-1 # (s,g,a) pairs, the last action is removed
        self.init_att = init_att
        self.freeze_att = freeze_att
        for k,v in kwargs.items():
            setattr(self, k, v)

class SingleAttention(nn.Module):
    ''' A simplified decision transformer, with single layer of attention, no drop-out or layer normalization'''
    def __init__(self, config):
        '''
        config, a class containing at least following configurations of the model:\n
        - n_embd, int, dimension of token embeddings
        - seq_length, int, maximum input token sequence length, used to create buffer
        '''
        super().__init__()
        self.n_embd = config.n_embd 
        # self.n_out = config.vocab # output dimension

        # Key, Query, Value layers. Remove bias to be consistent with paper
        self.key = nn.Linear(self.n_embd, self.n_embd)
        self.query = nn.Linear(self.n_embd, self.n_embd)
        self.value = nn.Linear(self.n_embd, self.n_embd)

        self.register_buffer("mask", torch.tril(torch.ones(config.seq_length, config.seq_length))
                                .view(1, config.seq_length, config.seq_length))

        # Ouput decoder layer
        # self.out_proj = nn.Linear(self.n_embd, self.n_out)

    def forward(self, tokens):
        '''
        Input: tokens: tensor of size (batch, context length, self.n_embd)\n, input token embeddings 0,1,..., ctx-1
        Output: tensor of size (batch, context length, self.n_embd). Can be regarded as a probability encoding of tokens 1,2,...,ctx.
        (If we take decoder over dim=-1, we get probability distibutions.)
        '''
        batch, real_seq_length, dim = tokens.size() # real_seq_length is the true input token sequence length, dim should be self.n_embd

        # attention matrices, all tensors (batch, ctx, dim). Only one head
        K = self.key(tokens)
        Q = self.query(tokens)
        V = self.value(tokens)
        # print(f"V={V}")

        # attention, follow the transformer paper
        att = (Q @ K.transpose(-2,-1)) * (1.0 / math.sqrt(dim)) # (batch, ctx, ctx)
        # logger.info(f"Attention shape {att.shape}, token shape {tokens.shape}")

        # Mask attention. 
        att = att.masked_fill(self.mask[:,:real_seq_length,:real_seq_length] == 0, float('-inf'))
        att = F.softmax(att, dim = -1) # row-wise softmax
        # print(f"att = {att}")
        output = att @ V # (batch, ctx, dim), unprojected output 

        # output projection
        # output = self.out_proj(out_att) # (batch, ctx, self.n_out)
        return output
    
    def init_freeze_params(self, init_value = None, freeze = True):
        '''
        Initialize attention parameters, and freeze the parameters.
        - init_value: None | float. If None, then do not initialize. If float, then init all paramters to that value
        - freeze: bool. If True, then all parameters in Attention don't require grad.
        '''
        logger.info(f"Init_value {init_value}")
        if init_value is not None:
            for param in self.parameters():
                param.data.fill_(init_value)

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

class SimpleDT(nn.Module):
    '''DT with SingleAttention'''
    def __init__(self, config):
        '''
        config: contains 
        - state_dim: int, dimension of state 
        - n_act: int, number of possible actions. Actions are given in one-hot representation
        - n_embd: int, dimension of token (state, action, rtg) embedding 
        - horizon: int, horizon of the MDP
        - init_att = None: float or None. If float, then init attention params with init_att
        - freeze_att = False: bool, if True, then freeze attention params' gradients
        '''
        super().__init__()
        self.state_dim = config.state_dim
        self.action_dim = config.n_act # In one-hot representation, action_dim == n_act
        self.n_embd = config.n_embd
        self.horizon = config.horizon

        # encoder
        self.embed_timestep = nn.Embedding(self.horizon, self.n_embd)
        self.embed_return = nn.Linear(1, self.n_embd, bias=True)
        self.embed_state = nn.Linear(self.state_dim, self.n_embd, bias=True)
        self.embed_action = nn.Linear(self.action_dim, self.n_embd, bias=True)

        # Transformer, here we use the SingleAttention layer
        self.transformer = SingleAttention(config)
        
        # init and freeze attention layer
        logger.info(f"Init_freeze: config.init_att {type(config.init_att)}")
        self.transformer.init_freeze_params(init_value = config.init_att, freeze = config.freeze_att)

        # decoder, do not take softmax here. It will be taken implicitly in F.CrossEntropy, and in get_action
        # self.decode_action = nn.Sequential(*([nn.Linear(self.n_embd, self.action_dim)] + [nn.Softmax(dim = -1)]))
        self.decode_action = nn.Linear(self.n_embd, self.action_dim, bias=True)

    def forward(self, states, actions, rtgs, timesteps):
        '''
        - states: (batch, ctx, state_dim)
        - actions: (batch, ctx-1, action_dim). 
        - rtgs: (batch, ctx, 1)
        - timesteps: (batch, ctx)
        Output: (batch, ctx, action_dim), the logits of predicted actions for all steps in ctx. 
        It is a probability distribution over actions, after softmax is taken
        '''

        # print(f"Action: {actions.shape}, timesteps {timesteps.shape}")
        batch, ctx = states.shape[0], states.shape[1] # store batch_size and ctx_length

        # logger.info(f"timesteps: {timesteps}")
        time_embeddings = self.embed_timestep(timesteps.type(torch.int64)) # (batch, ctx, n_embd)
        # logger.info(f"time_embeddings: {time_embeddings.shape}")

        # Convert to float32 to avoid type mismatch bugs
        state_embeddings = self.embed_state(states.type(torch.float32)) + time_embeddings
        # logger.info(f"state_embeddings: {state_embeddings.shape}")

        # First pad the last action, to make torch.stack work
        actions_pad = torch.cat([actions, torch.zeros(actions.shape[0], 1, actions.shape[2])], dim=1)
        # print(f"Pad action: {actions_pad.shape}")
        action_embeddings = self.embed_action(actions_pad.type(torch.float32))
        # print(f"action_embedding: {action_embeddings.shape}")
        # padding 0 to make torch.stack work, but we don't want the last ctx
        # action_embeddings = torch.cat([action_embeddings, torch.zeros((action_embeddings.shape[0],1,action_embeddings.shape[2]))], dim=1)
        # print(f"action_embedding cat: {action_embeddings.shape}")
        action_embeddings = action_embeddings + time_embeddings # (batch, ctx, action_dim)
        # print(f"action_embedding + time: {action_embeddings.shape}")
        

        rtg_embeddings = self.embed_return(rtgs.type(torch.float32)) + time_embeddings
        # print(f"rtg embeddings: {rtg_embeddings.shape}")

        stacked_inputs = torch.stack((state_embeddings, rtg_embeddings, action_embeddings), dim=1) # (batch, 3, ctx, n_embd)
        # print(f"Stacked inputs: {stacked_inputs.shape}")
        stacked_inputs = stacked_inputs.permute(0,2,1,3).reshape(batch, 3*ctx, self.n_embd)
        stacked_inputs = stacked_inputs[:,:-1,:] # Throw away the last padding action
        # print(f"Transformer input: {stacked_inputs.shape}")
        att_out = self.transformer(stacked_inputs) # (batch, 3*ctx-1, n_embd)

        actions_hidden = att_out[:, 1::3, :] # (batch, ctx, n_embd), keep all actions hidden state
        pred_action_logits = self.decode_action(actions_hidden) # (batch, ctx, action_dim)

        return pred_action_logits
        




# Test the model
# class AttentionConfig:
#     def __init__(self):
#         self.n_embd = 2

# config = AttentionConfig()
# model = SingleAttention(config)
# layer = nn.Linear(2,2,bias=False, )
# x = torch.randn(3,3,2)
# print(x)
# print(layer(x))



