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

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    # max_epochs = 10
    # batch_size = 64
    # learning_rate = 3e-4
    # betas = (0.9, 0.95)
    # grad_norm_clip = 1.0
    # weight_decay = 0.1 # only applied on matmul weights
    # # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    # lr_decay = False
    # warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    # final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # # checkpoint settings
    # ckpt_prefix = None
    # num_workers = 0 # for DataLoader
    # horizon = 5
    # desired_rtg = horizon
    # env = None # gym.Env class, the MDP environment

    def __init__(self, 
                 batch_size, 
                 num_workers, 
                 grad_norm_clip, 
                 max_epochs, 
                 desired_rtg, 
                 ckpt_prefix, 
                 env, 
                 eval_repeat, 
                 ctx_length, 
                 horizon,
                 lr = 6e-3,
                 weight_decay = 0.1, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grad_norm_clip = grad_norm_clip
        self.max_epochs = max_epochs
        self.desired_rtg = desired_rtg
        self.ckpt_prefix = ckpt_prefix
        self.env = env
        self.eval_repeat = eval_repeat
        self.ctx_length = ctx_length
        self.horizon = horizon
        self.learning_rate = lr
        self.weight_decay = weight_decay
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, dataset, config):
        '''
        model: nn.Module, should be class SimpleDT \n
        dataset: torch.utils.data.Dataset, should be training dataset \n
        config: TrainerConfig class, contains following elements:
        - batch_size, int
        - num_workers, int, for dataloader
        - grad_norm_clip
        - max_epochs
        - desired_rtg
        - ckpt_prefix
        - env
        - eval_repeat
        - ctx_length
        - learning_rate, float, for optimizer
        - weight_decay, float, for optimizer
        - horizon, int, horizon of the env
        '''
        self.model = model
        self.dataset = dataset
        # Initiatilize the configurations directly in __init__
        self.batch_size = config.batch_size 
        self.num_workers = config.num_workers # For dataloader
        self.grad_norm_clip = config.grad_norm_clip # For GD, avoid exploding gradients
        self.max_epochs = int(config.max_epochs) # number of training epochs
        self.desired_rtg = config.desired_rtg
        self.ckpt_prefix = config.ckpt_prefix # Prefix of checkpoint path, to save the model
        self.env = config.env # gym.Env class, the MDP environment to evaluate performance
        self.eval_repeat = int(config.eval_repeat) # int, times of repeated evaluation (take average on the returns received)
        self.ctx_length = int(config.ctx_length) # int, context length of training data
        self.horizon = int(config.horizon)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), 
                                           lr = config.learning_rate, 
                                           weight_decay = config.weight_decay)

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            print(f"device={self.device}")
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def _loss(self, pred_policy, true_action):
        '''
        Compute the cross-entropy loss.
        - pred_policy: (batch, ctx, action_dim), prob. distribution of the predicted action
        - true_action: (batch, ctx, action_dim), the true action in one-hot representation. Can also be a prob. distribution
        Return: (batch*ctx)
        '''
        # Must be converted to shape (N,C), N is batch*ctx, C is action_dim / num_actions
        return F.cross_entropy(pred_policy.reshape(-1, pred_policy.shape[-1]), true_action.reshape(-1, true_action.shape[-1]))
        

    def _run_epoch(self, epoch_num):
        '''
        Run one epoch in the training process \n
        Epoch_num: int, epoch number, used to display in progress bar
        '''

        loader = DataLoader(self.dataset, shuffle=True, pin_memory=True,
                            batch_size= self.batch_size,
                            num_workers= self.num_workers)
        
        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (states, actions, rtgs, timesteps) in pbar:
            '''
            states, (batch, ctx, state_dim)
            actions, (batch, ctx, action_dim)
            rtgs, (batch, ctx, rtg_dim)
            timesteps, (batch, ctx)
            '''                
            states = states.to(self.device)
            actions = actions.to(self.device)
            rtgs = rtgs.to(self.device)
            timesteps = timesteps.to(self.device)
            # print(f"Training it {it}, actions = {actions}, rtgs = {rtgs}, timesteps = {timesteps}")
            # print("Finish loading data to devices.")
            # print(f"Training epoch: actions {actions.shape}, states {states.shape}, rtgs {rtgs.shape}")
            
            # forward the model
            with torch.set_grad_enabled(True):
                history_actions = actions[:, :-1, :] # The last action is not input
                # target_action = actions[:,-1,:] # The last action is target
                pred_action_logits = self.model(states, history_actions, rtgs, timesteps)
                loss = self._loss(pred_action_logits, actions) # (batch, ctx)

                loss = loss.mean() # scalar tensor. Collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                # print("Finish loss computation.")      
            
            self.model.zero_grad()
            loss.backward()
            # print(f"Gradient of K: {self.model.transformer.key.weight.grad}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()

            pbar.set_description(f"Epoch {epoch_num+1}, iter {it}: train loss {loss.item():.5f}.")

    def eval(self, desired_rtg):
        self.model.train(False)
        rets = [] # list of returns achieved in each epoch
        for epoch in range(self.eval_repeat):
            states = self.env.reset()
            states = states.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) # (1,1,state_dim)
            rtgs = torch.Tensor([[[desired_rtg]]]).to(self.device) # (1,1,1)
            timesteps = torch.Tensor([[0]]) # (1,1)
            
            # Initialize action
            action_dim = self.env.get_num_action() # Get the number of possible actions
            actions = torch.empty((1,0,action_dim)) # Actions are represented in one-hot

            # print(f"Eval forward: states {states.shape}, actions {actions.shape}")

            ret = 0 # total return 
            for h in range(self.horizon):
                # Get action
                pred_action_logits = self.model(states, actions, rtgs, timesteps) #(1, ctx, action_dim)
                pred_action_logits = pred_action_logits[:, -1, :] # keep the last step hidden_state
                # print(f"Eval logits {pred_action_logits[0,:]}")
                probs = F.softmax(pred_action_logits[0,:], dim=0) #(action_dim)
                # print(f"Step {h+1}, eval policy {probs}")
                sample = torch.multinomial(probs, num_samples=1).item() # int, between [0,action_dim-1]
                sample_action = torch.zeros(action_dim)
                sample_action[sample] = 1 # one-hot representation, (action_dim)

                # Observe next states, rewards,
                next_state, reward, _ = self.env.step(sample_action) # (state_dim), scalar

                # Calculate return
                ret += reward
                
                # Update states, actions, rtgs, timesteps
                next_state = next_state.unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,state_dim)
                states = torch.cat([states, next_state], dim=1)
                states = states[:, -self.ctx_length: , :] # truncate to ctx_length

                sample_action = sample_action.unsqueeze(0).unsqueeze(0).to(self.device)
                actions = torch.cat([actions, sample_action], dim=1)
                actions = actions[:, -self.ctx_length+1: , :] # actions length is ctx-1

                next_rtg = rtgs[0,0,-1] - reward
                next_rtg = next_rtg * torch.ones(1,1,1) # (1,1,1)
                rtgs = torch.cat([rtgs, next_rtg], dim=1)
                rtgs = rtgs[:, -self.ctx_length: , :]

                # Update timesteps
                timesteps = torch.cat([timesteps, (h+1)*torch.ones(1,1)], dim = 1) 
                # timesteps = torch.cat([timesteps, next_timestep], dim=1)
                timesteps = timesteps[:, -self.ctx_length: ]
            # Add the ret to list
            rets.append(ret)
        # Compute average ret
        avg_ret = sum(rets) / self.eval_repeat
        print(f"target return: {desired_rtg}, eval return: {avg_ret}")
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

        self.eval(self.desired_rtg)
        for epoch in range(self.max_epochs):
            print(f"------------\nEpoch {epoch+1}")
            self._run_epoch(epoch)
            eval_return = self.eval(self.desired_rtg)
            if abs(eval_return-self.desired_rtg) < np.abs(best_return-self.desired_rtg):
                best_return = eval_return
                best_epoch = epoch
                if self.ckpt_prefix is not None:
                    epoch_ckpt_path = self.ckpt_prefix + f"_best.pth"
                    print(f"Better return {best_return}, better epoch {best_epoch}. Save model to {epoch_ckpt_path}")
                    self._save_checkpoint(epoch_ckpt_path)
        # if self.ckpt_prefix is not None:
        #     epoch_ckpt_path = self.ckpt_prefix + f"_final.pth"
        #     print(f"Final return {best_return}, final epoch {best_epoch}. Save model to {epoch_ckpt_path}")
        #     self._save_checkpoint(epoch_ckpt_path)
