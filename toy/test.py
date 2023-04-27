# import gym
import numpy as np
import torch
import torch.nn.functional as F

from env.no_best_RTG import BanditEnv

HORIZON = 20
# MDP = BanditEnv(H = HORIZON)

# for i in range(HORIZON):
#     action = np.random.randint(0,2)
#     MDP.render()
#     print(f"Take action {action}")
#     MDP.step(action=action)
# MDP.render()
# print("Finish")

device = 'cpu'

model = torch.load('./model/m5_epoch49.pth', map_location=device)
model.train(False)
# model = torch.load('./model/model4_s3_epoch0.pth')
env = BanditEnv(horizon = HORIZON)

# timestep 1
states = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
actions = torch.Tensor([[]]).type(torch.int64).unsqueeze(0)
rtgs = torch.Tensor([[20]]).type(torch.int64).unsqueeze(0)
timesteps = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
states = states.to(device)
actions = actions.to(device)
rtgs = rtgs.to(device)
timesteps = timesteps.to(device)
logits, _ = model(states, actions, targets = None, rtgs=rtgs,timesteps=timesteps)
# print(logits)
# print(logits.shape)
logits = logits[:, -1, :]
# print(logits)
probs = F.softmax(logits, dim=-1)
print(f"Timestep 1, prob1 is {probs[0,1]}")

# timestep 2,...,20
states = torch.Tensor([[0],[0]]).type(torch.int64).unsqueeze(0)
actions = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
states = states.to(device)
actions = actions.to(device)

# In coding, t starts from 0
for t in range(1,20):

    rtgs = torch.Tensor([[20-t+1],[20-t]]).unsqueeze(0)
    timesteps = torch.Tensor([[t]]).type(torch.int64).unsqueeze(0)

    rtgs = rtgs.to(device)
    timesteps = timesteps.to(device)


    logits, _ = model(states, actions, targets = None, rtgs=rtgs,timesteps=timesteps)
    # print(logits)
    # print(logits.shape)
    logits = logits[:, -1, :]
    # print(logits)
    probs = F.softmax(logits, dim=-1)
    print(f"Timestep {t+1}, prob1 is {probs[0,1]}")




# ix = torch.multinomial(probs, num_samples=1)
# print(ix)
# args=Args(horizon=self.config.horizon)
# env = Env(self.config.horizon)
# env.eval()

# T_rewards, T_Qs = [], []
# done = True
# for i in range(10): # Run the Env 10 times to take average
#     state = env.reset()
#     # state = torch.Tensor([state])
#     state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0) # size (batch,block_size,1)
#     rtgs = [ret]
#     # first state is from env, first rtg is target return, and first timestep is 0
#     sampled_action = sample(self.model.module, state, 1, temperature=1.0, sample=True, actions=None, 
#         rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
#         timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

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

#         state = state.unsqueeze(0).unsqueeze(0).to(self.device)

#         all_states = torch.cat([all_states, state], dim=0)

#         rtgs += [rtgs[-1] - reward]
#         # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
#         # timestep is just current timestep
#         sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
#             actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
#             rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
#             timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))
# env.close()
# eval_return = sum(T_rewards)/10.
# print("target return: %d, eval return: %d" % (ret, eval_return))
# # logger.info("target return: %d, eval return: %d", ret, eval_return)
# self.model.train(True)
# return eval_return

    
    
