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

from torch.utils.tensorboard import SummaryWriter  

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
                 ckpt_prefix, 
                 env, 
                 eval_repeat, 
                 horizon,
                 lr = 6e-3,
                 weight_decay = 0.1, 
                 r_scale = 1.0, **kwargs):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.grad_norm_clip = grad_norm_clip
        self.max_epochs = max_epochs
        self.ckpt_prefix = ckpt_prefix
        self.env = env
        self.eval_repeat = eval_repeat
        self.horizon = horizon
        self.learning_rate = lr
        self.weight_decay = weight_decay
        self.r_scale = r_scale
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
        - ckpt_prefix
        - env
        - eval_repeat
        - learning_rate, float, for optimizer
        - weight_decay, float, for optimizer
        - horizon, int, horizon of the env
        - tradeoff_coef, alpha in CQL paper.
        - tb_log, path to tb log directory
        - r_scale, scale the reward for better performance
        '''
        self.model = model
        self.dataset = dataset
        self.config = config

        self.action_space = config.env.get_action_space() # Tensor(num_action, action_dim), all possible actions
        
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

    # def _loss(self, pred_policy, true_action):
    #     '''
    #     Compute the cross-entropy loss.
    #     - pred_policy: (batch, ctx, action_dim), prob. distribution of the predicted action
    #     - true_action: (batch, ctx, action_dim), the true action in one-hot representation. Can also be a prob. distribution
    #     Return: (batch*ctx)
    #     '''
    #     # Must be converted to shape (N,C), N is batch*ctx, C is action_dim / num_actions
    #     return F.cross_entropy(pred_policy.reshape(-1, pred_policy.shape[-1]), true_action.reshape(-1, true_action.shape[-1]))
    
    def _max_q(self, states, timesteps):
        '''
        Compute the maximum Q-function under given states, used for bellman operator. \n
        Special case: when timesteps == horizon, need to manually output 0
        Input: states, (batch,state_dim); timesteps, (batch, )
        Output: (batch,)
        '''
        max_qs = []
        for i in range(states.shape[0]): # Compute max for each batch element
            action_space = self.action_space # (num_action, action_dim)
            num_action = action_space.shape[0] # number of possible actions

            state = states[i] # (state_dim)
            state = state.repeat(num_action, 1) # (num_action, state_dim)

            timestep = timesteps[i]
            if timestep == self.config.horizon: # terminal step, Q function is 0
                max_qs += [0]
            else:
                timestep = timestep.repeat(num_action) # (num_action)
                qfs = self.model(state,action_space,timestep)
                max_q = torch.max(qfs).item() # scalar
                max_qs += [max_q] # add max_q to list
        return torch.tensor(max_qs)
    
    def _get_optimal_action(self, states, timesteps, env, record_qf = False, epoch=-1):
        '''
        Compute the optimal action under given states. \n
        Input: 
        - states, (batch,state_dim); 
        - timesteps, (batch, )
        - env, the environment to sample action. Should be self.config.env
        - record_qf: If true, use tensorboard to record the q-functions at timestep 0
        - epoch: Epoch of training. Only used when record_qf is True.
        Output: (batch,action_dim)
        '''
        action_space = self.action_space # (num_action, action_dim)
        num_action = action_space.shape[0] # number of possible actions
        action_dim = action_space.shape[1]
        opt_actions = torch.zeros((0,action_dim))
        for i in range(states.shape[0]): # Compute max for each batch element

            state = states[i] # (state_dim)
            state = state.repeat(num_action, 1) # (num_action, state_dim)

            timestep = timesteps[i]
            timestep = timestep.repeat(num_action) # (num_action)
            
            state = state.type(torch.float32).to(self.device)
            all_actions = action_space.type(torch.float32).to(self.device)
            timestep = timestep.to(self.device)
            qfs = self.model(state,all_actions,timestep)

            # Record in tb
            # if record_qf:
            #     tb_scalars = {} # dict to be added to tb_writer
            #     for j in range(num_action):
            #         # assert state.shape[1]==1, f"State dim {state.shape[1]} larger than 1"
            #         key = f"a{j}"
            #         tb_scalars[key] = qfs[j] * self.config.r_scale # Store the unscaled qf
            #     self.tb_writer.add_scalars(f"Qf-t{int(timestep[j].item())}", tb_scalars, epoch)

            # if record_qf:
            #     q13 = qfs[13].item()
            #     sort_qfs, idx = torch.sort(qfs, descending=True)
            #     print(f"Timestep {timestep[0].item()}, state {state[0].item()}, Q-function best {sort_qfs[0].item()}, best action {idx[0].item()}, Q13 {q13}")

            # opt_index = torch.argmax(qfs).item()
            opt_action = env.get_action(qfs, mode='best')
            # if timestep[0].item() == 0:
            #     print(f"Timestep {timestep[0].item()}, take action {opt_action}")

            opt_action = opt_action.reshape(1,-1) # (1,action_dim)
            # opt_action = action_space[opt_index].unsqueeze(0) # (1,action_dim)
            opt_actions = torch.cat([opt_actions, opt_action],dim=0)
        return opt_actions
    
    def _log_sum_exp_q(self, states, timesteps):
        '''
        Compute $log \sum_a exp(Q(s,a,t))$ \n
        Input: states, (batch,state_dim); timesteps, (batch, )
        Output: (batch,)
        '''
        result_qs = []
        for i in range(states.shape[0]): # Compute max for each batch element
            action_space = self.action_space # (num_action, action_dim)
            num_action = action_space.shape[0] # number of possible actions

            state = states[i] # (state_dim)
            state = state.repeat(num_action, 1) # (num_action, state_dim)

            timestep = timesteps[i]
            timestep = timestep.repeat(num_action) # (num_action)
            qfs = self.model(state,action_space,timestep) # (num_action,)
            result_q = torch.logsumexp(qfs,dim=0).item()
            result_qs += [result_q] # add max_q to list
        return torch.tensor(result_qs)


    def _run_epoch(self, epoch_num, r_scale=1.0):
        '''
        Use CQL(H), Q-learning variant \n
        Run one epoch in the training process \n
        Epoch_num: int, epoch number, used to display in progress bar. \n
        r_scale: float, scale the reward received
        '''

        loader = DataLoader(self.dataset, shuffle=self.config.shuffle, pin_memory=True,
                            batch_size= self.config.batch_size,
                            num_workers= self.config.num_workers,)
        
        # losses = []
        pbar = tqdm(enumerate(loader), total=len(loader))
        for it, (states, actions, rewards, next_states, timesteps) in pbar:
            '''
            states, (batch, state_dim)
            actions, (batch, action_dim)
            rewards, (batch, 1) or (batch,) ?
            timesteps, (batch)
            next_states, (batch, state_dim)
            '''                
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.reshape(-1) / self.config.r_scale # guarantee shape (batch), scale the reward
            timesteps = timesteps.type(torch.int).to(self.device)

            # print(f"run_epoch: batch = {states.shape[0]}")

            # forward the model
            with torch.set_grad_enabled(True):
                qfs = self.model(states,actions,timesteps) # estimated Q-functions Q(s,a), (batch,)
                qfs = qfs.to('cpu') # move back to cpu
                # print(f"qfs {qfs.device}")
                with torch.no_grad():
                    max_next_qfs = self._max_q(next_states,timesteps+1) #(batch, )
                # print(f"max_next_qfs: {max_next_qfs.device}")
                    bell_qfs = rewards + max_next_qfs # bellman operator, (batch, )
                assert bell_qfs.requires_grad == False, "bell_qfs still requires grad!"
                # print(f"bell_qfs: {bell_qfs.device}")
                # print(f"bell_qfs.shape is {bell_qfs.shape}, max_next_qfs {max_next_qfs.shape}, rewards {rewards.shape}")
                bell_loss = 0.5 * F.mse_loss(qfs, bell_qfs) # conventional Q-learning loss

                logsumexp_qfs = self._log_sum_exp_q(states, timesteps) # (batch,)
                ood_penalty = torch.mean(logsumexp_qfs) # o.o.d penalty term

                in_distr_bonus = torch.mean(qfs) # in distribution encouraging term

                loss = self.config.tradeoff_coef * (ood_penalty - in_distr_bonus) + bell_loss

                loss = loss.mean() # scalar tensor. Collapse all losses if they are scattered on multiple gpus

                if self.config.tb_log is not None:
                    self.tb_writer.add_scalar('training_loss', loss.item(), epoch_num)
                # losses.append(loss.item())
                # print("Finish loss computation.")      
            
            self.model.zero_grad()
            loss.backward()
            # print(f"Gradient of K: {self.model.transformer.key.weight.grad}")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
            self.optimizer.step()

            pbar.set_description(f"Epoch {epoch_num+1}, iter {it}: train loss {loss.item():.5f}.")

    def eval(self, train_epoch=-1):
        '''
        train_epoch, int, the epoch of training. -1 for initial value.
        Used for tensorboard.
        '''
        self.model.train(False)

        # Log parameters, for one-hot encoding, single layer only
        # model_module = self.model.module if hasattr(self.model, 'module') else self.model
        # linear_net = model_module.network.network[0]
        # weight = linear_net.weight.data[0,:] # Should be Tensor(4)
        # bias = linear_net.bias.data[0] # should be Tensor(scalar)
        # # assert weight.shape[0] == 4, f"Invalid weight shape {weight.shape}"
        # # assert bias.shape[0] == 1, f"Invalid bias shape {bias.shape}"
        # param_dict = {'s':weight[0].item(),'wa0':weight[1].item(), 'wa1':weight[2].item(), 'wt':weight[-1].item(), 'b':bias.item()}
        # self.tb_writer.add_scalars('params',param_dict, train_epoch)

        rets = [] # list of returns achieved in each epoch
        env = self.config.env
        for epoch in range(self.config.eval_repeat):
            state = env.reset() #(state_dim)
            state = state.type(torch.float32).to(self.device).unsqueeze(0) # (1,state_dim)
            timestep = torch.Tensor([0]).type(torch.int) #(1,)
            
            # Initialize action

            # print(f"Eval forward: states {states.shape}, actions {actions.shape}")

            ret = 0 # total return 
            for h in range(self.config.horizon):
                action = self._get_optimal_action(state,timestep,env,record_qf=True,epoch=train_epoch)
                # print(f"Epoch {train_epoch}, timestep {h}, action {action}")
                # action = action.item() # Change to int, only for 1-dim actions!

                # Update state, observe reward
                state, reward, _ = env.step(action) # (state_dim), scalar
                state = state.unsqueeze(0) # (1, state_dim)

                # Calculate return
                ret += reward
                timestep += 1
            # Add the ret to list
            rets.append(ret)
        # Compute average ret
        avg_ret = sum(rets) / self.config.eval_repeat
        print(f"Eval return: {avg_ret}")
        if self.config.tb_log is not None:
            self.tb_writer.add_scalar("avg_ret", avg_ret, train_epoch)
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

        self.eval(train_epoch=-1)
        for epoch in range(self.config.max_epochs):
            print(f"------------\nEpoch {epoch+1}")
            self._run_epoch(epoch)
            self.eval(train_epoch=epoch)
            # if eval_return-self.desired_rtg) < np.abs(best_return-self.desired_rtg):
            #     best_return = eval_return
            #     best_epoch = epoch
            #     if self.ckpt_prefix is not None:
            #         epoch_ckpt_path = self.ckpt_prefix + f"_best.pth"
            #         print(f"Better return {best_return}, better epoch {best_epoch}. Save model to {epoch_ckpt_path}")
            #         self._save_checkpoint(epoch_ckpt_path)
        if self.config.ckpt_prefix is not None:
            epoch_ckpt_path = self.config.ckpt_prefix + f"_final.pth"
            print(f"Save final model to {epoch_ckpt_path}")
            self._save_checkpoint(epoch_ckpt_path)
