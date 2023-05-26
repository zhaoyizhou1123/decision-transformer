# import gym
import numpy as np
import torch
import torch.nn.functional as F

from env.no_best_RTG import BanditEnv as Env
from mingpt.utils import sample
from mingpt.trainer_toy import TrainerConfig, Trainer
from torch.utils.tensorboard import SummaryWriter  
import argparse
import os

parser = argparse.ArgumentParser()
# parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--context_length', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
# parser.add_argument('--num_steps', type=int, default=500000)
# parser.add_argument('--num_buffers', type=int, default=50)
# parser.add_argument('--game', type=str, default='Breakout')
parser.add_argument('--batch_size', type=int, default=1)
# 
# parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
parser.add_argument('--data_file', type=str, default='./dataset/toy.csv')
parser.add_argument('--log_level', type=str, default='WARNING')
parser.add_argument('--goal', type=int, default=5, help="The desired RTG")
parser.add_argument('--horizon', type=int, default=5, help="Should be consistent with dataset")
parser.add_argument('--ckpt_prefix', type=str, default=None )
parser.add_argument('--rate', type=float, default=6e-3, help="learning rate of Trainer" )
parser.add_argument('--hash', type=bool, default=False, help="Hash states if True")
parser.add_argument('--tb_path', type=str, default="./logs/", help="Folder to tensorboard logs" )
parser.add_argument('--tb_suffix', type=str, default="0", help="Suffix used to discern different runs" )
args = parser.parse_args()
print(args)

HORIZON = args.horizon
# MDP = BanditEnv(H = HORIZON)

# for i in range(HORIZON):
#     action = np.random.randint(0,2)
#     MDP.render()
#     print(f"Take action {action}")
#     MDP.step(action=action)
# MDP.render()
# print("Finish")

device = 'cpu'

env = Env(horizon = HORIZON)

# states = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
# actions = torch.Tensor([[]]).type(torch.int64).unsqueeze(0)
# timesteps = torch.Tensor([[0]]).type(torch.int64).unsqueeze(0)
# states = states.to(device)
# actions = actions.to(device)
# timesteps = timesteps.to(device)

# Focus on step 1
# for epoch in [15,20,25,30,35,40,45]:
#     model_path = f"./model/m5_rate4_epoch{epoch}.pth"
#     model = torch.load(model_path, map_location=device)
#     model.train(False)
#     # model = torch.load('./model/model4_s3_epoch0.pth')
#     for desired_rtg in [0,5,10,15,20,25,30,35,40]:
#         rtgs = torch.Tensor([[desired_rtg]]).type(torch.int64).unsqueeze(0)
#         rtgs = rtgs.to(device)
#         logits, _ = model(states, actions, targets = None, rtgs=rtgs,timesteps=timesteps)
#         # print(logits)
#         # print(logits.shape)
#         logits = logits[:, -1, :]
#         # print(logits)
#         probs = F.softmax(logits, dim=-1)
#         print(f"Epoch {epoch}, desired rtg is {desired_rtg}, timestep 1, prob1 is {probs[0,1]}")
    
model_path = args.ckpt_prefix+"_final.pth"
model = torch.load(model_path, map_location=device)
model.train(False)
desired_rtg = 20
# cur_rtg = 20


data_file = args.data_file[10:-4] # Like "toy5", "toy_rev". args.data_file form "./dataset/xxx.csv"
tb_dir = f"{data_file}_ctx{args.context_length}_batch{args.batch_size}_goal{args.goal}_lr{args.rate}_{args.tb_suffix}"
tb_dir_path = os.path.join(args.tb_path,tb_dir)
# os.makedirs(tb_dir_path, exist_ok=True) # The directory must exist

tconf = TrainerConfig(max_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.rate,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*10*args.context_length*3,
                      num_workers=4, model_type=args.model_type, max_timestep=args.horizon, horizon=args.horizon, 
                      desired_rtg=args.goal, ckpt_prefix = args.ckpt_prefix, env = env, tb_log = tb_dir_path, 
                      ctx = args.context_length)
trainer = Trainer(model, None, None, tconf)

tb_writer = SummaryWriter(tb_dir_path)
for goal in range(20,21):
    eval_ret = trainer.get_returns(goal, is_debug=True)
    tb_writer.add_scalar('eval_ret', eval_ret, goal)


# For ctx=20
# def get_returns(ret):
#     model.train(False)
#     # args=Args(horizon=self.config.horizon)
#     env = Env(HORIZON)
#     # env.eval()

#     T_rewards, T_Qs = [], []
#     done = True
#     state = env.reset()
#     # state = torch.Tensor([state])
#     state = state.type(torch.float32).to(device).unsqueeze(0).unsqueeze(0) # size (batch,block_size,1)
#     rtgs = [ret]
#     # first state is from env, first rtg is target return, and first timestep is 0
#     sampled_action = sample(model, state, 1, temperature=1.0, sample=True, actions=None, 
#         rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
#         timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(device))
#     print(f"Evaluation, timestep 1, action={sampled_action}")
#     # j = 0
#     all_states = state
#     actions = []
#     reward_sum = 0
#     for j in range(1,20):
#         # print(f"Step = {step_cnt}, done is {done}")
#         # if done:
#         #     state, reward_sum, done = env.reset(), 0, False
#         action = sampled_action.cpu().numpy()[0,-1]
#         actions += [sampled_action]
#         state, reward, done = env.step(action)
#         reward_sum += reward
#         # j += 1

#         # if done:
#         #     T_rewards.append(reward_sum)
#         #     break

#         state = state.unsqueeze(0).unsqueeze(0).to(device)

#         all_states = torch.cat([all_states, state], dim=0)

#         rtgs += [rtgs[-1] - reward]
#         # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
#         # timestep is just current timestep
#         sampled_action = sample(model, all_states.unsqueeze(0), 1, temperature=1.0, sample=True, 
#             actions=torch.tensor(actions, dtype=torch.long).to(device).unsqueeze(1).unsqueeze(0), 
#             rtgs=torch.tensor(rtgs, dtype=torch.long).to(device).unsqueeze(0).unsqueeze(-1), 
#             timesteps=(j * torch.ones((1, 1, 1), dtype=torch.int64).to(device)))
#         print(f"Evaluation, timestep {j+1}, action={action}")
#     env.close()

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


# for prev_action in range(1,2):
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

    
    
