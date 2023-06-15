# Sample a dataset, store in a csv file

# from env.bandit_env import BanditEnv as Env
from env.time_var_env import TimeVarEnv as Env
# from env.no_best_RTG import BanditEnvReverse as Env
from env.utils import sample
from dataset.utils import FullPolicy
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default='./dataset/timevar_exp.csv')
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--env_path', type=str, default='./env/env_hard.txt')
parser.add_argument('--time_depend_s', action='store_true')
parser.add_argument('--time_depend_a', action='store_true')
# parser.add_argument('--num_trajectories', type=int, default=100)
args = parser.parse_args()

# Create Env instance
# env = Env(args.env_path, sample, time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)
env = Env(horizon=args.horizon, num_actions=10)
horizon = env.get_horizon()
num_actions = env.get_num_action()

# Some common policies
full_policies = FullPolicy(horizon)
all1_policy = full_policies.all_good_policy()
all0_policy = full_policies.all_bad_policy()
all_rand_policy = full_policies.all_rand_policy()
alt10_policy = full_policies.alt_good_bad_policy() # 1,0,1,0,...
alt01_policy = full_policies.alt_bad_good_policy() # 0,1,0,1,...

first0then1_policy = full_policies.bad_good_policy(1)
first1then0_policy = full_policies.good_bad_policy(1)
# all1last0_policy = full_policies.good_bad_policy(100)

def sample_and_write(dataset_path: str, env, horizon, sample_policy_list: list, num_repeat_list: list, 
                     time_depend_a: bool):
    '''
    - dataset_path: Path to output file
    - env: Sampling environment
    - horizon: Env environment
    - sample_policy_list: list of full sampling policy, each element is list(utils.Policy)
    - num_repeat_list: how much times each policy in sample_policy_list is repeated. Must be of same length 
    - time_depend: bool. If true, action becomes (a+1)*(t+1).
    as sample_policy_list.
    '''
    assert len(sample_policy_list) == len(num_repeat_list), "Mismatch lengths!"
    with open(dataset_path, "w") as f:
        # Write labels
        for i in range(horizon):
            f.write(f"t{i},s{i},a{i},r{i},")
        f.write(f"t{horizon}\n")
        for policy_idx in range(len(sample_policy_list)):
            sample_policy = sample_policy_list[policy_idx]
            assert len(sample_policy) == horizon, f"Wrong sample policy length {len(sample_policy)}!"
            repeat_times = num_repeat_list[policy_idx]
            for repeat in range(repeat_times):
                # Start sampling
                state = env.reset()
                for h in range(horizon):
                    f.write(f"{h},{int(state.item())},")
                    stage_policy = sample_policy[h]
                    action = stage_policy.sample_action()
                    if time_depend_a:
                        real_num_action = int(env.get_num_action() / horizon)
                        action = action + h * real_num_action
                    state, reward, _ = env.step(torch.tensor(action))
                    f.write(f"{action},{reward:.1f},")
                f.write(f"{horizon}\n")

# sample_policy_list = [all1_policy, all0_policy, alt01_policy, alt10_policy]
# sample_policy_list = [alt01_policy, alt10_policy, all1_policy, all0_policy]
# sample_policy_list = [full_policies.repeat_action_policy(i) for i in range(num_actions)]
sample_policy_list = [full_policies.double_loop_full_policy(i,j,num_actions) for j in range(num_actions) for i in range(num_actions)]

# num_repeat_list = [1 for _ in range(num_actions)]
num_repeat_list = [1 for _ in sample_policy_list]

sample_and_write(args.output_file, env, horizon, sample_policy_list, num_repeat_list, args.time_depend_a)


# with open(args.output_file, "w") as f:
#     # Write annotation
#     for i in range(horizon):
#         f.write(f"t{i},s{i},a{i},r{i},")
#     f.write(f"t{horizon}\n")
#     single_trajectory_num = args.num_trajectories//4
#     # Sample. 
#     # for epoch in range(args.num_trajectories):
#     #     state = env.reset()
#     #     for h in range(horizon):
#     #         f.write(f"{h},{int(state.item())},")
#     #         action = rand_policy.sample_action()
#     #         state, reward, _ = env.step(torch.tensor(action))
#     #         f.write(f"{action},{int(reward)},")
#     #     f.write(f"{horizon}\n")


#     # Expert data
#     for epoch in range(0,single_trajectory_num):
#         # 5 steps good, then all bad
#         state = env.reset()
#         f.write(f"0,{int(state.item())},")
#         action = good_policy.sample_action()
#         state, reward, _ = env.step(torch.tensor(action))
#         f.write(f"{action},{int(reward)},")
#         for h in range(1,horizon):
#             f.write(f"{h},{int(state.item())},")
#             # if h % 2 == 0:
#             action = good_policy.sample_action()
#             # else:
#             #     action = bad_policy.sample_action()
#             state, reward, _ = env.step(torch.tensor(action))
#             f.write(f"{action},{int(reward)},")
#         f.write(f"{horizon}\n")
#     for epoch in range(single_trajectory_num, 2*single_trajectory_num):
#         # 5 steps good, then all bad
#         state = env.reset()
#         f.write(f"0,{int(state.item())},")
#         action = good_policy.sample_action()
#         state, reward, _ = env.step(torch.tensor(action))
#         f.write(f"{action},{int(reward)},")
#         for h in range(1,horizon):
#             f.write(f"{h},{int(state.item())},")
#             # if h % 2 == 0:
#             action = bad_policy.sample_action()
#             # else:
#             #     action = good_policy.sample_action()
#             state, reward, _ = env.step(torch.tensor(action))
#             f.write(f"{action},{int(reward)},")
#         f.write(f"{horizon}\n")
#     for epoch in range(2*single_trajectory_num, 3*single_trajectory_num):
#         # 5 steps good, then all bad
#         state = env.reset()
#         f.write(f"0,{int(state.item())},")
#         action = bad_policy.sample_action()
#         state, reward, _ = env.step(torch.tensor(action))
#         f.write(f"{action},{int(reward)},")
#         for h in range(1,horizon):
#             f.write(f"{h},{int(state.item())},")
#             action = bad_policy.sample_action()
#             state, reward, _ = env.step(torch.tensor(action))
#             f.write(f"{action},{int(reward)},")
#         f.write(f"{horizon}\n")
#     for epoch in range(2*single_trajectory_num, 3*single_trajectory_num):
#         # 5 steps good, then all bad
#         state = env.reset()
#         f.write(f"0,{int(state.item())},")
#         action = bad_policy.sample_action()
#         state, reward, _ = env.step(torch.tensor(action))
#         f.write(f"{action},{int(reward)},")
#         for h in range(1,horizon):
#             f.write(f"{h},{int(state.item())},")
#             action = good_policy.sample_action()
#             state, reward, _ = env.step(torch.tensor(action))
#             f.write(f"{action},{int(reward)},")
#         f.write(f"{horizon}\n")
    
#         for h in range(10):       
#             f.write(f"{h},{int(state.item())},")
#             # Always run the best policy
#             action = good_policy.sample_action()
#             state, reward, _ = env.step(torch.tensor(action))
#             f.write(f"{action},{int(reward)},")    
#         for h in range(10, horizon):
#             # state = env.get_state()
#             f.write(f"{h},{int(state.item())},")
#             # Always run the worst policy
#             action = bad_policy.sample_action()
#             state, reward, _ = env.step(torch.tensor(action))
#             f.write(f"{action},{int(reward)},")
#         f.write(f"{horizon}\n")

        # good,bad,good,...
        # for h in range(20):
        #     state = env.get_state()
        #     f.write(f"{h},{int(state.item())},")
        #     if h % 2 == 0:
        #         action = good_policy.sample_action()
        #     else:
        #         action = bad_policy.sample_action()
        #     _, reward, _ = env.step(action)
        #     f.write(f"{action},{int(reward)},")
        # f.write(f"{horizon+1}\n")
    # for epoch in range(args.num_trajectories//2, args.num_trajectories):
    #     # 5 steps bad, then all good
    #     for h in range(10):
    #         state = env.get_state()
    #         f.write(f"{h},{int(state.item())},")
    #         # Always run the best policy
    #         action = bad_policy.sample_action()
    #         _, reward, _ = env.step(action)
    #         f.write(f"{action},{int(reward)},")    
    #     for h in range(10, horizon):
    #         state = env.get_state()
    #         f.write(f"{h},{int(state.item())},")
    #         # Always run the worst policy
    #         action = good_policy.sample_action()
    #         _, reward, _ = env.step(action)
    #         f.write(f"{action},{int(reward)},")
    #     f.write(f"{horizon}\n")

        # bad,good,bad,...
        # for h in range(20):
        #     state = env.get_state()
        #     f.write(f"{h},{int(state.item())},")
        #     if h % 2 == 1:
        #         action = good_policy.sample_action()
        #     else:
        #         action = bad_policy.sample_action()
        #     _, reward, _ = env.step(action)
        #     f.write(f"{action},{int(reward)},")
        # f.write(f"{horizon+1}\n")

        

