import numpy as np
import torch
import time
from tqdm import tqdm
import wandb
import os
from typing import Optional

from torch.utils.data.dataloader import DataLoader
from maze.algos.decision_transformer.models.decision_transformer import DecisionTransformer
from maze.utils.dataset import TrajCtxDataset


import torch.nn.functional as F


class SequenceTrainer:
    def __init__(self, config, model: DecisionTransformer, offline_dataset: TrajCtxDataset, rollout_dataset: Optional[TrajCtxDataset] = None):
        '''
        offline_trajs / rollout_trajs: List[Trajectory]
        config members:
        - batch_size
        - lr
        - device
        '''
        self.config = config
        self.device = self.config.device
        self.model = model.to(self.device)
        self.batch_size = config.batch_size
        # self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.offline_dataset = offline_dataset
        self.rollout_dataset = rollout_dataset

        warmup_steps = 10000
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda steps: min((steps+1)/warmup_steps, 1)
        )
        self.start_time = time.time()

        if self.config.ckpt_path is not None:
            os.makedirs(self.config.ckpt_path, exist_ok=True)

    def loss_fn(self, pred_action, true_action):
        '''
        Compute the MSE loss.
        - pred_action: (batch, action_dim), logits of the predicted action (don't do softmax)
        - true_action: (batch, action_dim), the true action in 1-dim representation
        Return: scalar tensor. The mean of each loss
        '''
        # print(f"Trainer_mlp 127: true_action {true_action.shape}, pred_action {pred_action.shape}")
        return F.mse_loss(pred_action, true_action)

    # def eval_fns():
    #     pass

    # def train_iteration(self, num_steps, iter_num=0, print_logs=False):

    #     train_losses = []
    #     logs = dict()

    #     train_start = time.time()

    #     self.model.train()
    #     for _ in range(num_steps):
    #         train_loss = self.train_step()
    #         train_losses.append(train_loss)
    #         if self.scheduler is not None:
    #             self.scheduler.step()

    #     logs['time/training'] = time.time() - train_start

    #     eval_start = time.time()

    #     self.model.eval()
    #     for eval_fn in self.eval_fns:
    #         outputs = eval_fn(self.model)
    #         for k, v in outputs.items():
    #             logs[f'evaluation/{k}'] = v

    #     logs['time/total'] = time.time() - self.start_time
    #     logs['time/evaluation'] = time.time() - eval_start
    #     logs['training/train_loss_mean'] = np.mean(train_losses)
    #     logs['training/train_loss_std'] = np.std(train_losses)

    #     for k in self.diagnostics:
    #         logs[k] = self.diagnostics[k]

    #     if print_logs:
    #         print('=' * 80)
    #         print(f'Iteration {iter_num}')
    #         for k, v in logs.items():
    #             print(f'{k}: {v}')

    #     return logs

    # def train_step(self):
    #     states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
    #     action_target = torch.clone(actions)

    #     state_preds, action_preds, reward_preds = self.model.forward(
    #         states, actions, rewards, rtg[:,:-1], timesteps, attention_mask=attention_mask,
    #     )

    #     act_dim = action_preds.shape[2]
    #     action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
    #     action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

    #     loss = self.loss_fn(
    #         None, action_preds, None,
    #         None, action_target, None,
    #     )

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
    #     self.optimizer.step()

    #     with torch.no_grad():
    #         self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()

    #     return loss.detach().cpu().item()

    def eval(self, desired_rtg, train_epoch):
        '''
        state_mean/std: Used for state normalization. Get from offline_dataset only currently
        '''
        state_mean, state_std = self.offline_dataset.get_normalize_coef()
        self.model.train(False)
        rets = [] # list of returns achieved in each epoch
        env = self.config.env
        action_dim = env.action_space.shape[0]
        for epoch in range(self.config.eval_repeat):
            states, _ = env.reset()
            if hasattr(env, 'get_true_observation'): # For pointmaze
                states = env.get_true_observation(states)
            states = torch.from_numpy(states)
            states = states.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) # (1,1,state_dim)
            rtgs = torch.Tensor([[[desired_rtg]]]).to(self.device) # (1,1,1)
            timesteps = torch.Tensor([[0]]).to(self.device) # (1,1)
            
            # Initialize action
            actions = torch.empty((1,0,action_dim)).to(self.device) # Actions are represented in one-hot

            # print(f"Eval forward: states {states.shape}, actions {actions.shape}")

            ret = 0 # total return 
            for h in range(self.config.horizon):
                # Get action
                pred_action = self.model.get_action((states - state_mean) / state_std,
                                                      actions.type(torch.float32),
                                                      rtgs.type(torch.float32),
                                                      timesteps.type(torch.float32)) # (act_dim)
                # pred_actions = self.model(states, actions, rtgs, timesteps) #(1, action_dim)
                # pred_action = pred_actions[0,:] # (action_dim)
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
                    print(f"Step {h+1}, action is {pred_action.detach().cpu()}, observed next state {next_state}, reward {reward}")   
                next_state = torch.from_numpy(next_state)
                # Calculate return
                ret += reward
                
                # Update states, actions, rtgs, timesteps
                next_state = next_state.unsqueeze(0).unsqueeze(0).to(self.device) # (1,1,state_dim)
                states = torch.cat([states, next_state], dim=1)
                states = states[:, -self.config.ctx: , :] # truncate to ctx_length

                pred_action = pred_action.unsqueeze(0).unsqueeze(0).to(self.device) # (1, 1, action_dim)
                
                if self.config.ctx > 1:
                    # print(actions.shape, pred_action.shape)
                    actions = torch.cat([actions, pred_action], dim=1)
                    actions = actions[:, -self.config.ctx+1: , :] # actions length is ctx-1
                # else ctx = 1, actions is always 0

                next_rtg = rtgs[0,0,-1] - reward
                next_rtg = next_rtg * torch.ones(1,1,1).to(self.device) # (1,1,1)
                rtgs = torch.cat([rtgs, next_rtg], dim=1)
                rtgs = rtgs[:, -self.config.ctx: , :]

                # Update timesteps
                timesteps = torch.cat([timesteps, (h+1)*torch.ones(1,1).to(self.device)], dim = 1) 
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
   
    
    def _run_epoch(self, epoch_num):
        '''
        Run one epoch in the training process \n
        Epoch_num: int, epoch number, used to display in progress bar. \n
        During training, we convert action to one_hot_hash
        '''
        if self.rollout_dataset is None: # Only offline dataset, don't differ
            dataset = self.offline_dataset
        else:
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
        for it, (states, actions, _, rtgs, timesteps, attention_mask) in pbar:
            '''
            states, (batch, ctx, state_dim)
            actions, (batch, ctx, action_dim)
            rtgs, (batch, ctx, 1)
            timesteps, (batch, ctx)
            attention_mask, (batch, ctx)
            '''    

            states = states.type(torch.float32).to(self.device)
            actions = actions.type(torch.float32).to(self.device)
            rtgs = rtgs.type(torch.float32).to(self.device)
            timesteps = timesteps.to(self.device).long()
            attention_mask = attention_mask.to(self.device)

            action_target = torch.clone(actions)

            # forward the model
            state_preds, action_preds, reward_preds = self.model.forward(
                states, actions, rtgs, timesteps, attention_mask=attention_mask,
            )

            act_dim = action_preds.shape[2]
            action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            loss = self.loss_fn(
                action_preds,
                action_target
            )

            # self.optimizer.zero_grad()
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            # self.optimizer.step()

            # with torch.no_grad():
            #     self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
            
            self.model.zero_grad()
            loss.backward()
            # print(f"Gradient of K: {self.model.transformer.key.weight.grad}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()

            pbar.set_description(f"Epoch {epoch_num+1}, iter {it}: train loss {loss.item():.5f}.")

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
                    epoch_ckpt_path = os.path.join(self.config.ckpt_path, "output_dt_policy_best.pth")
                    print(f"Better return {best_return}, better epoch {best_epoch}. Save model to {epoch_ckpt_path}")
                    self._save_checkpoint(epoch_ckpt_path)

    def _save_checkpoint(self, ckpt_path):
        '''
        ckpt_path: str, path of storing the model
        '''
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(raw_model, ckpt_path)
