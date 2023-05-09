# Test the parameters of the model

import numpy as np
import torch
import torch.nn.functional as F

from env.no_best_RTG import BanditEnvOneHot as Env
from mingpt.utils import sample

# Basic configurations
HORIZON = 20
device = 'cpu'
env = Env(horizon = HORIZON)
model_path = f"./model/m5_simple_final.pth"
model = torch.load(model_path, map_location=device)
model.train(False)
ctx_length = 2

def eval(desired_rtg):
    model.train(False)
    rets = [] # list of returns achieved in each epoch
    for epoch in range(1):
        states = env.reset()
        states = states.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0) # (1,1,state_dim)
        rtgs = torch.Tensor([[[desired_rtg]]]).to(device) # (1,1,1)
        timesteps = torch.Tensor([[0]]) # (1,1)
        
        # Initialize action
        action_dim = env.get_num_action() # Get the number of possible actions
        actions = torch.empty((1,0,action_dim)) # Actions are represented in one-hot

        # print(f"Eval forward: states {states.shape}, actions {actions.shape}")

        ret = 0 # total return 
        for h in range(HORIZON):
            # Get action
            print(f"horizon {h}")
            pred_action_logits = model(states, actions, rtgs, timesteps) #(1, ctx, action_dim)
            pred_action_logits = pred_action_logits[:, -1, :] # keep the last step hidden_state
            # print(f"Eval logits {pred_action_logits[0,:]}")
            probs = F.softmax(pred_action_logits[0,:], dim=0) #(action_dim)
            print(f"prob1 is {probs[1]}")
            # print(f"Step {h+1}, eval policy {probs}")
            sample = torch.multinomial(probs, num_samples=1).item() # int, between [0,action_dim-1]
            sample_action = torch.zeros(action_dim)
            sample_action[sample] = 1 # one-hot representation, (action_dim)

            # Observe next states, rewards,
            next_state, reward, _ = env.step(sample_action) # (state_dim), scalar

            # Calculate return
            ret += reward
            
            # Update states, actions, rtgs, timesteps
            next_state = next_state.unsqueeze(0).unsqueeze(0).to(device) # (1,1,state_dim)
            states = torch.cat([states, next_state], dim=1)
            states = states[:, -ctx_length: , :] # truncate to ctx_length

            sample_action = sample_action.unsqueeze(0).unsqueeze(0).to(device)
            actions = torch.cat([actions, sample_action], dim=1)
            actions = actions[:, -ctx_length+1: , :] # actions length is ctx-1

            next_rtg = rtgs[0,0,-1] - reward
            next_rtg = next_rtg * torch.ones(1,1,1) # (1,1,1)
            rtgs = torch.cat([rtgs, next_rtg], dim=1)
            rtgs = rtgs[:, -ctx_length: , :]

            # Update timesteps
            timesteps = torch.cat([timesteps, (h+1)*torch.ones(1,1)], dim = 1) 
            # timesteps = torch.cat([timesteps, next_timestep], dim=1)
            timesteps = timesteps[:, -ctx_length: ]
        # Add the ret to list
        rets.append(ret)
    # Compute average ret
    avg_ret = sum(rets) / 1
    print(f"target return: {desired_rtg}, eval return: {avg_ret}")
    # Set the model back to training mode
    model.train(True)
    return avg_ret

eval(20)