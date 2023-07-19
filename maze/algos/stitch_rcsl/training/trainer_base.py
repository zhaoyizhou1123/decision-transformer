'''
Basic trainer model
'''

"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
import wandb
import os

from torch.utils.tensorboard import SummaryWriter  

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    # max_epochs = 10
    # batch_size = 64
    # learning_rate = 3e-4
    # betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    # lr_decay = False
    # warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    # final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # # checkpoint settings
    ckpt_path = None
    num_workers = 1 # for DataLoader
    tb_log = None
    # r_loss_weight = 0.5 # weight of r_loss w.r.t (r_loss + s_loss)
    # horizon = 5
    # desired_rtg = horizon
    # env = None # gym.Env class, the MDP environment

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class BaseTrainer:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        # self.action_space = config.env.get_action_space() # Tensor(num_action, action_dim), all possible actions

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            print(f"device={self.device}")

        if config.tb_log is not None:
            self.tb_writer = SummaryWriter(config.tb_log)
        
        # model_module = self.model.module if hasattr(self.model, 'module') else self.model
        # self.tb_writer.add_graph(model_module, (torch.tensor([0]),torch.tensor([1,0]),torch.tensor(0)))

    def _loss(self, pred, truth, weight = None):
        '''
        Compute weighted 2-norm loss
        - pred: (batch, dim) / dim
        - true: (batch, dim) / dm
        - weight: (batch, ) | None. None means no weight
        Return: scalar tensor. The mean of each loss
        '''
        # print(f"Trainer_mlp 127: true_action {true_action.shape}, pred_action {pred_action.shape}")
        # return F.mse_loss(pred, truth)

        # mse_loss of each batch element
        batch_loss = torch.norm(pred-truth, dim = -1) # (batch)

        # weighted average
        if weight is None:
            return torch.mean(batch_loss)
        else:
            return torch.sum(weight * batch_loss) / torch.sum(weight)
            


    def _run_epoch(self, epoch_num):
        '''
        Run one epoch in the training process \n
        Epoch_num: int, epoch number, used to display in progress bar. \n
        During training, we convert action to one_hot_hash
        '''

        return None
    
    def _save_checkpoint(self, ckpt_path):
        '''
        ckpt_path: str, dir of storing dynamics, behavior policy, and init_state model
        '''
        # DataParallel wrappers keep raw model object in .module attribute
        # raw_dynamics_model = self.dynamics_model.module if hasattr(self.dynamics_model, "module") else self.dynamics_model
        # raw_policy_model = self.behavior_policy_model.module if hasattr(self.behavior_policy_model, "module") else self.behavior_policy_model
        # raw_init_model = self.init_state_model.module if hasattr(self.init_state_model, "module") else self.init_state_model

        # # d_path = f"{ckpt_prefix}_dynamics.pth" # dynamics model
        # # p_path = f"{ckpt_prefix}_behavior.pth" # behavior model
        # d_path = os.path.join(ckpt_path, "dynamics.pth")
        # p_path = os.path.join(ckpt_path, "behavior.pth")
        # i_path = os.path.join(ckpt_path, "init.pth")
        # print(f"Saving dynamics model to {d_path}, behavior policy model to {p_path}, init_state model to {i_path}" )
        # torch.save(raw_dynamics_model, d_path)
        # torch.save(raw_policy_model, p_path) 
        # torch.save(raw_init_model, i_path)
        pass

    def train(self):
        min_loss = float('inf')
        for epoch in range(self.config.max_epochs):
            loss = self._run_epoch(epoch)
            # self.eval(self.config.desired_rtg, train_epoch=epoch)
            if loss < min_loss:
                min_loss = loss
                if self.config.ckpt_path is not None:
                    self._save_checkpoint(self.config.ckpt_path)
        # if self.config.ckpt_path is not None:
        #     self._save_checkpoint(self.config.ckpt_path)
