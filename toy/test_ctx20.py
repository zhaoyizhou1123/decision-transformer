# import gym
import numpy as np
import torch
import torch.nn.functional as F

from env.no_best_RTG import BanditEnv as Env
from mingpt.utils import sample

# Basic configurations
HORIZON = 20
device = 'cpu'
env = Env(horizon = HORIZON)
model_path = f"./model/m_alt_best.pth"
model = torch.load(model_path, map_location=device)
model.train(False)

# For ctx=20
def get_returns(ret, model, HORIZON):
    model.train(False)
    # args=Args(horizon=HORIZON)
    env = Env(HORIZON)
    # env.eval()

    T_rewards, T_Qs = [], []
    done = True
    for i in range(1): # Run the Env 10 times to take average
        state = env.reset()
        # state = torch.Tensor([state])
        state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0) # size (batch,block_size,1)
        rtgs = [ret]
        # first state is from env, first rtg is target return, and first timestep is 0
        model = model.module if hasattr(model, "module") else model # Avoid error: no attribute module
        sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
            rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
            timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device))
        print(f"Evaluation, timestep 1, action={sampled_action}")
        j = 0
        all_states = state
        actions = []
        step_cnt = 1 # Count timesteps
        while True:
            # print(f"Step = {step_cnt}, done is {done}")
            if done:
                state, reward_sum, done = env.reset(), 0, False
            action = sampled_action.cpu().numpy()[0,-1]
            actions += [sampled_action]
            state, reward, done = env.step(action)
            reward_sum += reward
            j += 1

            if done:
                T_rewards.append(reward_sum)
                break

            state = state.unsqueeze(0).unsqueeze(0).to(device)

            all_states = torch.cat([all_states, state], dim=0)

            rtgs += [rtgs[-1] - reward]
            # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
            # timestep is just current timestep
            sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
                actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
                timesteps=(min(j, HORIZON) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
            print(f"Step {j+1}, {sampled_action}")
    env.close()
    eval_return = sum(T_rewards)/1.
    print("target return: %d, eval return: %f" % (ret, eval_return))
    # logger.info("target return: %d, eval return: %d", ret, eval_return)
    # model.train(True)
    return eval_return


for rtg in range(20,21):
    # print(f"Desired rtg {rtg}")
    eval_ret = get_returns(rtg, model, HORIZON)
    print(f"Desired {rtg}, evaluate {eval_ret}\n -----------")
# get_returns(cur_rtg)

# # timestep 1
# state = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
# action = torch.Tensor([[1]]).type(torch.int64).unsqueeze(0)

# states = state
# actions = torch.zeros((1,1,1), dtype=torch.int64)
# timesteps = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
# rtgs = torch.Tensor([[cur_rtg]]).type(torch.int64).unsqueeze(0)

# for t in range(20):
#     logits, _ = model(states, actions, targets = None, rtgs=rtgs,timesteps=timesteps)
#     logits = logits[:, -1, :]
#     probs = F.softmax(logits, dim=-1)
#     print(f"Current desired rtg is {cur_rtg}, timestep {t+1}, prob1 is {probs[0,1]}")
#     # timestep 2,...,20
#     # states = torch.Tensor([[0],[0]]).type(torch.int64).unsqueeze(0)
#     states = torch.cat([states,state], dim=0)

#     # actions = torch.Tensor([[prev_action]]).type(torch.int64).unsqueeze(0)
#     actions = torch.cat([actions,action], dim=0)
#     timesteps[0,0,0] += 1

#     cur_rtg -= 1
#     rtgs = torch.cat([rtgs, torch.Tensor([[[cur_rtg]]])], dim = 0)

# # In coding, t starts from 0
# for t in range(1,20):

#     # rtgs = torch.Tensor([[desired_rtg-t+1],[desired_rtg-t]]).unsqueeze(0)
#     rtgs = torch.Tensor([[desired_rtg-t]]).unsqueeze(0)
#     timesteps = torch.Tensor([[t]]).type(torch.int64).unsqueeze(0)

#     rtgs = rtgs.to(device)
#     timesteps = timesteps.to(device)


#     logits, _ = model(states, actions, targets = None, rtgs=rtgs,timesteps=timesteps)
#     # print(logits)
#     # print(logits.shape)
#     logits = logits[:, -1, :]
#     # print(logits)
#     probs = F.softmax(logits, dim=-1)
#     print(f"Timestep {t+1}, prob1 is {probs[0,1]}")


# for prev_action in range(2):
#     # timestep 1
#     states = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
#     actions = torch.Tensor([[]]).type(torch.int64).unsqueeze(0)
#     timesteps = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
#     rtgs = torch.Tensor([[desired_rtg]]).type(torch.int64).unsqueeze(0)
#     # rtgs = rtgs.to(device)
#     # states = states.to(device)
#     # actions = actions.to(device)
#     # timesteps = timesteps.to(device)
#     logits, _ = model(states, actions, targets = None, rtgs=rtgs,timesteps=timesteps)
#     logits = logits[:, -1, :]
#     probs = F.softmax(logits, dim=-1)
#     print(f"desired rtg is {desired_rtg}, timestep 1, prob1 is {probs[0,1]}")
#     # timestep 2,...,20
#     states = torch.Tensor([[0],[0]]).type(torch.int64).unsqueeze(0)
#     # states = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
#     actions = torch.Tensor([[prev_action]]).type(torch.int64).unsqueeze(0)
#     # actions = torch.Tensor([]).type(torch.int64).unsqueeze(0)
#     # states = states.to(device)
#     # actions = actions.to(device)

#     # In coding, t starts from 0
#     for t in range(1,20):

#         rtgs = torch.Tensor([[desired_rtg-t+1],[desired_rtg-t+1-prev_action]]).unsqueeze(0)
#         # rtgs = torch.Tensor([[desired_rtg-t]]).unsqueeze(0)
#         timesteps = torch.Tensor([[t]]).type(torch.int64).unsqueeze(0)

#         # rtgs = rtgs.to(device)
#         # timesteps = timesteps.to(device)


#         logits, _ = model(states, actions, targets = None, rtgs=rtgs,timesteps=timesteps)
#         # print(logits)
#         # print(logits.shape)
#         logits = logits[:, -1, :]
#         # print(logits)
#         probs = F.softmax(logits, dim=-1)
#         print(f"Timestep {t+1}, prev_action {prev_action}, desired RTG {rtgs[0,0,0]} and {rtgs[0,1,0]}, prob1 is {probs[0,1]}")




# ix = torch.multinomial(probs, num_samples=1)
# print(ix)
# args=Args(horizon=HORIZON)
# env = Env(HORIZON)
# env.eval()

# T_rewards, T_Qs = [], []
# done = True
# for i in range(10): # Run the Env 10 times to take average
#     state = env.reset()
#     # state = torch.Tensor([state])
#     state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0) # size (batch,block_size,1)
#     rtgs = [ret]
#     # first state is from env, first rtg is target return, and first timestep is 0
#     sampled_action = sample(model.module, state, 1, temperature=1.0, sample=True, actions=None, 
#         rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
#         timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device))

#     j = 0
#     all_states = state
#     actions = []
#     step_cnt = 1 # Count timesteps
#     while True:
#         # print(f"Step = {step_cnt}, done is {done}")
#         if done:
#             state, reward_sum, done = env.reset(), 0, False
#         action = sampled_action.cpu().numpy()[0,-1]
#         actions += [sampled_action]
#         state, reward, done = env.step(action)
#         reward_sum += reward
#         j += 1

#         if done:
#             T_rewards.append(reward_sum)
#             break

#         state = state.unsqueeze(0).unsqueeze(0).to(device)

#         all_states = torch.cat([all_states, state], dim=0)

#         rtgs += [rtgs[-1] - reward]
#         # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
#         # timestep is just current timestep
#         sampled_action = sample(model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
#             actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
#             rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
#             timesteps=(min(j, config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
# env.close()
# eval_return = sum(T_rewards)/10.
# print("target return: %d, eval return: %d" % (ret, eval_return))
# # logger.info("target return: %d, eval return: %d", ret, eval_return)
# model.train(True)
# return eval_return

    
    
