'''
Train the MDP model as well as the behavior policy
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
    # horizon = 5
    # desired_rtg = horizon
    # env = None # gym.Env class, the MDP environment

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class MdpTrainer:
    def __init__(self, dynamics_model, behavior_policy_model, dataset, config):
        '''
        Train dynamics model and behavior policy
        - dataset, must use TrajNextObsDataset
        - config, containing attributes:
            - learning_rate
            - weight_dcay
            - tb_log
            - batch_size
            - num_workers = 1
            - log_to_wandb
            - grad_norm_clip = 1.0
            - max_epochs
            - ckpt_prefix = None
        '''
        self.dynamics_model = dynamics_model
        self.behavior_policy_model = behavior_policy_model
        self.dataset = dataset
        self.config = config

        # self.action_space = config.env.get_action_space() # Tensor(num_action, action_dim), all possible actions
        
        self.dynamics_optimizer = torch.optim.AdamW(self.dynamics_model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)
        
        self.policy_optimizer = torch.optim.AdamW(self.behavior_policy_model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            print(f"device={self.device}")
            self.dynamics_model = torch.nn.DataParallel(self.dynamics_model).to(self.device)
            self.behavior_policy_model = torch.nn.DataParallel(self.behavior_policy_model).to(self.device)

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

        loader = DataLoader(self.dataset, shuffle=True, pin_memory=True,
                            batch_size= self.config.batch_size,
                            num_workers= self.config.num_workers)
        
        # losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (states, actions, rewards, _, timesteps, next_states) in pbar:
            '''
            states, (batch, state_dim)
            actions, (batch, action_dim)
            rewards, (batch, 1)
            timesteps, (batch,)
            next_states, (batch, state_dim)
            '''    

            states = states.type(torch.float32).to(self.device)
            actions = actions.type(torch.float32).to(self.device)
            rewards = rewards.type(torch.float32).to(self.device)
            timesteps = timesteps.type(torch.float32).to(self.device)
            next_states = next_states.type(torch.float32).to(self.device)

            # forward the model
            with torch.set_grad_enabled(True):
                # history_actions = actions[:, :-1, :]
                # target_action = actions[:,-1,:] # The last action is target
                pred_rewards, pred_next_states = self.dynamics_model(states, actions, timesteps) # (batch,), (batch, state_dim)
                support_actions, support_probs = self.behavior_policy_model(states, timesteps=None) # (batch, n_support, action_dim), (batch, n_support)
                n_support = support_probs.shape[1]
                support_actions = support_actions.reshape(-1, support_actions.shape[-1]) # (batch * n_spport, action_dim)
                support_probs = support_probs.reshape(-1) # (batch * n_support)


                # print(target_action.min(), target_action.max(), pred_actions.shape[-1])
                r_loss = self._loss(pred_rewards, rewards.squeeze(-1)) # Tensor(scalar)       
                s_loss = self._loss(pred_next_states, next_states)
                a_loss = self._loss(support_actions, actions.repeat_interleave(n_support, dim=0), weight = support_probs) # Repeat actions to (batch * n_support, action_dim)

                dynamics_loss = r_loss + s_loss # Simply adding the two losses



                # loss = loss.mean() # scalar tensor. Collapse all losses if they are scattered on multiple gpus
                # print("Finish loss computation.")      
                if self.config.tb_log is not None:
                    self.tb_writer.add_scalar('r_loss', r_loss.item(), epoch_num)
                    self.tb_writer.add_scalar('s_loss', s_loss.item(), epoch_num)
                    self.tb_writer.add_scalar('a_loss', a_loss.item(), epoch_num)

                if self.config.log_to_wandb:
                    wandb.log({'loss': r_loss.item()})
                    wandb.log({'loss': s_loss.item()})
                    wandb.log({'loss': a_loss.item()})
            
            self.dynamics_model.zero_grad()
            self.behavior_policy_model.zero_grad()

            dynamics_loss.backward()
            a_loss.backward()
            # print(f"Gradient of K: {self.model.transformer.key.weight.grad}")
            torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), self.config.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(self.behavior_policy_model.parameters(), self.config.grad_norm_clip)
            self.dynamics_optimizer.step()
            self.policy_optimizer.step()

            pbar.set_description(f"Epoch {epoch_num+1}, iter {it}: Total loss {(dynamics_loss + a_loss).item():.5f}.")
    
    def _save_checkpoint(self, ckpt_path):
        '''
        ckpt_path: str, path prefix of storing dynamics and behavior policy model
        '''
        # DataParallel wrappers keep raw model object in .module attribute
        raw_dynamics_model = self.dynamics_model.module if hasattr(self.dynamics_model, "module") else self.dynamics_model
        raw_policy_model = self.behavior_policy_model.module if hasattr(self.behavior_policy_model, "module") else self.behavior_policy_model

        # d_path = f"{ckpt_prefix}_dynamics.pth" # dynamics model
        # p_path = f"{ckpt_prefix}_behavior.pth" # behavior model
        d_path = os.path.join(ckpt_path, "dynamics.pth")
        p_path = os.path.join(ckpt_path, "behavior.pth")
        print(f"Saving dynamics model to {d_path}, behavior policy model to {p_path}" )
        torch.save(raw_dynamics_model, d_path)
        torch.save(raw_policy_model, p_path) 

    def train(self):
        for epoch in range(self.config.max_epochs):
            self._run_epoch(epoch)
            # self.eval(self.config.desired_rtg, train_epoch=epoch)
            # if eval_return-self.desired_rtg) < np.abs(best_return-self.desired_rtg):
            #     best_return = eval_return
            #     best_epoch = epoch
            #     if self.ckpt_prefix is not None:
            #         epoch_ckpt_path = self.ckpt_prefix + f"_best.pth"
            #         print(f"Better return {best_return}, better epoch {best_epoch}. Save model to {epoch_ckpt_path}")
            #         self._save_checkpoint(epoch_ckpt_path)
        if self.config.ckpt_path is not None:
            self._save_checkpoint(self.config.ckpt_path)
