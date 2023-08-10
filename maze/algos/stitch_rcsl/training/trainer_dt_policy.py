"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
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

class Trainer:
    def __init__(self, config, model, offline_dataset, rollout_dataset=None):
        '''
        model: nn.Module, should be class SimpleDT \n
        offline_dataset: torch.utils.data.Dataset, offline training dataset, for pretraining
        rollout_dataset: data collected from rollout
        config: TrainerConfig class, contains following elements:
        - batch_size, int
        - num_workers, int, for dataloader
        - grad_norm_clip
        - max_epochs: total number of epochs
        - pre_epochs: pretraining epochs
        - ckpt_path
        - env
        - eval_repeat
        - learning_rate, float, for optimizer
        - weight_decay, float, for optimizer
        - horizon, int, horizon of the env
        - tradeoff_coef, alpha in CQL paper.
        - tb_log, path to tb log directory
        '''
        self.model = model
        self.offline_dataset = offline_dataset
        self.rollout_dataset = rollout_dataset
        self.config = config

        # self.action_space = config.env.get_action_space() # Tensor(num_action, action_dim), all possible actions
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            print(f"device={self.device}")
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if config.tb_log is not None:
            self.tb_writer = SummaryWriter(config.tb_log)
        
        # model_module = self.model.module if hasattr(self.model, 'module') else self.model
        # self.tb_writer.add_graph(model_module, (torch.tensor([0]),torch.tensor([1,0]),torch.tensor(0)))

    def _loss(self, pred_action, true_action):
        '''
        Compute the MSE loss.
        - pred_action: (batch, action_dim), logits of the predicted action (don't do softmax)
        - true_action: (batch, action_dim), the true action in 1-dim representation
        Return: scalar tensor. The mean of each loss
        '''
        # print(f"Trainer_mlp 127: true_action {true_action.shape}, pred_action {pred_action.shape}")
        return F.mse_loss(pred_action, true_action)


    def _run_epoch(self, epoch_num):
        '''
        Run one epoch in the training process \n
        Epoch_num: int, epoch number, used to display in progress bar. \n
        During training, we convert action to one_hot_hash
        '''
        if epoch_num < self.config.pre_epochs:
            dataset = self.offline_dataset
            if self.config.debug:
                print(f"Pretraining") 
        else:
            dataset = self.rollout_dataset
            if self.config.debug:
                print(f"Training on rollout data")
        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size= self.config.batch_size,
                            num_workers= self.config.num_workers)
        
        # losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (states, actions, _, rtgs, timesteps) in pbar:
            '''
            states, (batch, ctx, state_dim)
            actions, (batch, ctx, action_dim)
            rtgs, (batch, ctx, 1)
            timesteps, (batch, ctx)
            '''    

            states = states.type(torch.float32).to(self.device)
            actions = actions.type(torch.float32).to(self.device)
            rtgs = rtgs.type(torch.float32).to(self.device)
            timesteps = timesteps.type(torch.float32).to(self.device)

            # forward the model
            with torch.set_grad_enabled(True):
                history_actions = actions[:, :-1, :]
                target_action = actions[:,-1,:] # The last action is target
                pred_actions = self.model(states, history_actions, rtgs, timesteps) # (batch,num_action)
                # print(target_action.min(), target_action.max(), pred_actions.shape[-1])
                loss = self._loss(pred_actions, target_action) # Tensor(scalar)               

                # loss = loss.mean() # scalar tensor. Collapse all losses if they are scattered on multiple gpus
                # print("Finish loss computation.")      
                if self.config.tb_log is not None:
                    self.tb_writer.add_scalar('training_loss', loss.item(), epoch_num)

                if self.config.log_to_wandb:
                    wandb.log({'loss': loss.item()})
            
            self.model.zero_grad()
            loss.backward()
            # print(f"Gradient of K: {self.model.transformer.key.weight.grad}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()

            pbar.set_description(f"Epoch {epoch_num+1}, iter {it}: train loss {loss.item():.5f}.")

    def eval(self, desired_rtg, train_epoch):
        self.model.train(False)
        rets = [] # list of returns achieved in each epoch
        action_dim = 1 # Assume no hashing. One-hot is converted in model
        env = self.config.env
        for epoch in range(self.config.eval_repeat):
            states, _ = env.reset()
            if hasattr(env, 'get_true_observation'): # For pointmaze
                states = env.get_true_observation(states)
            states = torch.from_numpy(states)
            states = states.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) # (1,1,state_dim)
            rtgs = torch.Tensor([[[desired_rtg]]]).to(self.device) # (1,1,1)
            timesteps = torch.Tensor([[0]]) # (1,1)
            
            # Initialize action
            actions = torch.empty((1,0,action_dim)).to(self.device) # Actions are represented in one-hot

            # print(f"Eval forward: states {states.shape}, actions {actions.shape}")

            ret = 0 # total return 
            for h in range(self.config.horizon):
                # Get action
                pred_actions = self.model(states, actions, rtgs, timesteps) #(1, action_dim)
                pred_action = pred_actions[0,:] # (action_dim)
                # pred_action_logits = pred_action_logits[:, -1, :] # keep the last step hidden_state
                # print(f"Eval logits {pred_action_logits[0,:]}")
                # probs = F.softmax(pred_action_logits[0,:], dim=0) #(num_action)

                # if self.config.sample:
                #     sample_action = torch.multinomial(probs, num_samples=1) # Tensor (1,), between [0,num_action-1]
                # else:
                #     _, sample_action = torch.topk(probs, k=1, dim=-1) # Tensor(1,)   

                # Print output policy only for the first epoch   
                # sample_action = torch.zeros(action_dim)
                # sample_action[sample] = 1 # one-hot representation, (action_dim)

                # Observe next states, rewards,
                next_state, reward, terminated, _, _ = env.step(pred_action.detach().cpu().numpy()) # (state_dim), scalar
                if hasattr(env, 'get_true_observation'): # For pointmaze
                    next_state = env.get_true_observation(next_state)
                if epoch == 0 and self.config.debug:
                    print(f"Step {h+1}, action is {pred_action.detach().cpu()}, observed next state {next_state}")   
                next_state = torch.from_numpy(next_state)
                # Calculate return
                ret += reward
                
                # Update states, actions, rtgs, timesteps
                next_state = next_state.unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,state_dim)
                states = torch.cat([states, next_state], dim=1)
                states = states[:, -self.config.ctx: , :] # truncate to ctx_length

                pred_action = pred_action.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, action_dim)
                
                if self.config.ctx > 1:
                    actions = torch.cat([actions, pred_action], dim=1)
                    actions = actions[:, -self.config.ctx+1: , :] # actions length is ctx-1
                # else ctx = 1, actions is always 0

                next_rtg = rtgs[0,0,-1] - reward
                next_rtg = next_rtg * torch.ones(1,1,1).to(self.device) # (1,1,1)
                rtgs = torch.cat([rtgs, next_rtg], dim=1)
                rtgs = rtgs[:, -self.config.ctx: , :]

                # Update timesteps
                timesteps = torch.cat([timesteps, (h+1)*torch.ones(1,1)], dim = 1) 
                # timesteps = torch.cat([timesteps, next_timestep], dim=1)
                timesteps = timesteps[:, -self.config.ctx: ]

                # if terminated: # Already reached goal, the rest steps get reward 1, break
                #     ret += self.config.horizon - 1 - h
                #     break
            # Add the ret to list
            rets.append(ret)
        # Compute average ret
        avg_ret = sum(rets) / self.config.eval_repeat
        print(f"target return: {desired_rtg}, eval return: {avg_ret}")
        if self.config.tb_log is not None:
            self.tb_writer.add_scalar("avg_ret", avg_ret, train_epoch)
        if self.config.log_to_wandb:
            wandb.log({"avg_ret": avg_ret})
        # Set the model back to training mode
        self.model.train(True)
        return avg_ret
    
    def _save_checkpoint(self, ckpt_path):
        '''
        ckpt_path: str, path of storing the model
        '''
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", ckpt_path)
        torch.save(raw_model, ckpt_path)

    def train(self):
        best_return = -float('inf')
        best_epoch = -1
        print(f"------------\nEpoch {0} (Initial model)")

        # Initialize and freeze the model

        self.eval(self.config.desired_rtg, train_epoch=-1)
        for epoch in range(self.config.max_epochs):
            print(f"------------\nEpoch {epoch+1}")
            self._run_epoch(epoch)
            eval_return = self.eval(self.config.desired_rtg, train_epoch=epoch)
            if eval_return > best_return:
                best_return = eval_return
                best_epoch = epoch
                if self.config.ckpt_path is not None:
                    epoch_ckpt_path = os.path.join(self.config.ckpt_path, "output_policy_best.pth")
                    print(f"Better return {best_return}, better epoch {best_epoch}. Save model to {epoch_ckpt_path}")
                    self._save_checkpoint(epoch_ckpt_path)
        # if self.config.ckpt_path is not None:
        #     epoch_ckpt_path = self.config.ckpt_path + f"_final.pth"
        #     print(f"Save final model to {epoch_ckpt_path}")
        #     self._save_checkpoint(epoch_ckpt_path)
