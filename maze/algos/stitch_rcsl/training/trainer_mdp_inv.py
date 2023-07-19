'''
Learn the model by inverse dynamics
'''
from .trainer_base import BaseTrainer
import math
import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
import wandb
import os

from torch.utils.tensorboard import SummaryWriter  

class MdpInvTrainer(BaseTrainer):
    def __init__(self, inv_dynamics_model, reward_model, next_state_model, init_state_model, dataset, config):
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
            - r_loss_weight = 0.5, the weight of r_loss w.r.t s_loss
        '''
        super().__init__(dataset, config)
        self.inv_dynamics_model = inv_dynamics_model
        self.reward_model = reward_model
        self.next_state_model = next_state_model
        self.init_state_model = init_state_model

        # self.action_space = config.env.get_action_space() # Tensor(num_action, action_dim), all possible actions
        
        self.dynamics_optimizer = torch.optim.AdamW(self.inv_dynamics_model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)
        
        self.policy_optimizer = torch.optim.AdamW(self.next_state_model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)
        
        self.init_optimizer = torch.optim.AdamW(self.init_state_model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)
        self.reward_optimizer = torch.optim.AdamW(self.reward_model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)

        if self.device is not 'cpu':
            self.inv_dynamics_model = torch.nn.DataParallel(self.inv_dynamics_model).to(self.device)
            self.reward_model = torch.nn.DataParallel(self.reward_model).to(self.device)
            self.next_state_model = torch.nn.DataParallel(self.next_state_model).to(self.device)
            self.init_state_model = torch.nn.DataParallel(init_state_model).to(self.device)

        if config.tb_log is not None:
            self.tb_writer = SummaryWriter(config.tb_log)
        
        # model_module = self.model.module if hasattr(self.model, 'module') else self.model
        # self.tb_writer.add_graph(model_module, (torch.tensor([0]),torch.tensor([1,0]),torch.tensor(0)))
        
    def _init_state_loss(self, pred_init_states, batch_states, batch_timesteps, pred_probs):
        '''
        Loss for init_state model
        pred_init_states: (n_support, state_dim)
        batch_states: (batch, state_dim)
        batch_timesteps: (batch, )
        pred_probs: (n_support)
        '''
        # batch = batch_states.shape[0]
        valid_idxs = (batch_timesteps == 0) # Choose init timesteps
        valid_truth_states = batch_states[valid_idxs] # (valid_batch, state_dim)
        valid_batch = valid_truth_states.shape[0]
        n_support = pred_init_states.shape[0]

        expand_pred_init_states = pred_init_states.repeat(valid_batch,1) # (n_support * valid_batch, state_dim)
        expand_valid_truth_states = valid_truth_states.repeat_interleave(n_support, dim=0) # (n_support * valid_batch, state_dim)
        expand_pred_probs = pred_probs.repeat(valid_batch) # (n_support * valid_batch)

        return self._loss(expand_pred_init_states, expand_valid_truth_states, expand_pred_probs)
            
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
        losses = []
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
                pred_actions = self.inv_dynamics_model(states, next_states, timestep=None) # (batch, action_dim)
                pred_rewards = self.reward_model(states,actions, timestep=None)

                support_next_states, support_probs = self.next_state_model(states, timesteps=None) # (batch, n_support, state_dim), (batch, n_support)
                n_support = support_probs.shape[1]
                support_next_states = support_next_states.reshape(-1, support_next_states.shape[-1]) # (batch * n_spport, state_dim)
                support_probs = support_probs.reshape(-1) # (batch * n_support)

                pred_init_states, pred_init_probs = self.init_state_model(timesteps) # (n_support, state_dim), n_support
                # pred_init_states = pred_init_states
                # pred_init_probs = pred_init_probs


                # print(target_action.min(), target_action.max(), pred_actions.shape[-1])
                r_loss = self._loss(pred_rewards, rewards.squeeze(-1)) # Tensor(scalar)     
                # print(next_states.repeat_interleave(n_support, dim=0).shape)
                # print(n_support)
                # print(support_probs)
                s_loss = self._loss(support_next_states, next_states.repeat_interleave(n_support, dim=0), weight = support_probs) # Repeat actions to (batch * n_support, action_dim)
                a_loss = self._loss(pred_actions, actions)

                # dynamics_loss = self.config.r_loss_weight * r_loss + (1-self.config.r_loss_weight) * s_loss # Simply adding the two losses

                init_loss = self._init_state_loss(pred_init_states, states, timesteps, pred_init_probs)

                # loss = loss.mean() # scalar tensor. Collapse all losses if they are scattered on multiple gpus
                # print("Finish loss computation.")      
                if self.config.tb_log is not None:
                    self.tb_writer.add_scalar('r_loss', r_loss.item(), epoch_num)
                    self.tb_writer.add_scalar('s_loss', s_loss.item(), epoch_num)
                    self.tb_writer.add_scalar('a_loss', a_loss.item(), epoch_num)
                    self.tb_writer.add_scalar('init_loss', init_loss.item(), epoch_num)

                if self.config.log_to_wandb:
                    wandb.log({'r_loss': r_loss.item()})
                    wandb.log({'s_loss': s_loss.item()})
                    wandb.log({'a_loss': a_loss.item()})
                    wandb.log({'init_loss': init_loss.item()})
            
            self.inv_dynamics_model.zero_grad()
            self.reward_model.zero_grad()
            self.next_state_model.zero_grad()
            self.init_state_model.zero_grad()

            r_loss.backward()
            s_loss.backward()
            a_loss.backward()
            init_loss.backward()
            # print(f"Gradient of K: {self.model.transformer.key.weight.grad}")
            torch.nn.utils.clip_grad_norm_(self.inv_dynamics_model.parameters(), self.config.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(self.next_state_model.parameters(), self.config.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(self.init_state_model.parameters(), self.config.grad_norm_clip)
            torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), self.config.grad_norm_clip)
            self.dynamics_optimizer.step()
            self.policy_optimizer.step()
            self.init_optimizer.step()
            self.reward_optimizer.step()

            pbar.set_description(f"Epoch {epoch_num+1}, iter {it}: NextState loss {s_loss.item():.3f}, Reward loss {r_loss.item():.3f}, Action loss {a_loss.item():.3f}, Init loss{init_loss.item():.3f}.")

            # don't calculate init_loss, as it contains (nan)
            losses.append((s_loss + r_loss + a_loss).item())

        return sum(losses) / len(losses)
    
    def _save_checkpoint(self, ckpt_path):
        '''
        ckpt_path: str, dir of storing dynamics, behavior policy, and init_state model
        '''
        # DataParallel wrappers keep raw model object in .module attribute
        raw_inv_dynamics_model = self.inv_dynamics_model.module if hasattr(self.inv_dynamics_model, "module") else self.inv_dynamics_model
        raw_reward_model = self.reward_model.module if hasattr(self.inv_dynamics_model, "module") else self.reward_model
        raw_next_state_model = self.next_state_model.module if hasattr(self.next_state_model, "module") else self.next_state_model
        raw_init_model = self.init_state_model.module if hasattr(self.init_state_model, "module") else self.init_state_model

        # d_path = f"{ckpt_prefix}_dynamics.pth" # dynamics model
        # p_path = f"{ckpt_prefix}_behavior.pth" # behavior model
        d_path = os.path.join(ckpt_path, "inv_dynamics.pth")
        p_path = os.path.join(ckpt_path, "next_state.pth")
        r_path = os.path.join(ckpt_path, "reward.pth")
        i_path = os.path.join(ckpt_path, "init.pth")
        print(f"Saving dynamics model to {d_path}, behavior policy model to {p_path}, init_state model to {i_path}" )
        torch.save(raw_inv_dynamics_model, d_path)
        torch.save(raw_next_state_model, p_path) 
        torch.save(raw_reward_model, r_path)
        torch.save(raw_init_model, i_path)