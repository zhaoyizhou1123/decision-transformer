# Sample a dataset, store in a csv file

from env.bandit_env import BanditEnv
from env.time_var_env import TimeVarEnv, TimeVarBanditEnv
# from env.no_best_RTG import BanditEnvReverse as Env
from env.utils import sample
import dataset.utils as utils
from dataset.utils import FullPolicy
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default='./dataset/linearq_exp.csv')
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--env_path', type=str, default='./env/env_linearq.txt')
parser.add_argument('--time_depend_s', action='store_true')
parser.add_argument('--time_depend_a', action='store_true')
parser.add_argument('--env_type', type=str, default='timevar_bandit', help='bandit or timevar or timevar_bandit or linearq')
# parser.add_argument('--num_trajectories', type=int, default=100)
args = parser.parse_args()
print(args)

# Create Env instance
# env = Env(args.env_path, sample, time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)

if args.env_type == 'timevar':
    env = TimeVarEnv(horizon=args.horizon, num_actions=10)
elif args.env_type == 'bandit':
    env = BanditEnv(args.env_path, sample, time_depend_s=args.time_depend_s, time_depend_a=args.time_depend_a)
elif args.env_type == 'linearq':
    env = BanditEnv(args.env_path, sample, mode='linearq')
elif args.env_type == 'timevar_bandit':
    env = TimeVarBanditEnv(horizon=args.horizon, num_actions=args.horizon)
else:
    raise Exception(f"Unimplemented env_type {args.env_type}!")
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
# first0then1_policy = full_policies.bad_good_policy(10)
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
        rets = [] # record returns
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

                rets.append(env._return)
        print(rets)

def env_hard_policy(horizon):
    assert horizon==20, f"Invalid horizon {horizon}!"
    sample_policy_list = []

    # s1 = 2
    for a in [0,2,4,6,8,10,12,14,16,18]:
        sample_policy_list += [full_policies.repeat_action_policy(a)]

        t2to19policy = utils.consecutive_stage_policy(utils.DeterministicPolicy(a), horizon-2)
        t0to1policy = [utils.DeterministicPolicy(a),utils.DeterministicPolicy(a+1)]
        full_policy = utils.concat_long_policy(t0to1policy,t2to19policy)
        sample_policy_list += [full_policy]

    # s1 = 3
    for a in [1,3,5,7,9,11,15,17,19]:
        sample_policy_list += [full_policies.repeat_action_policy(a)]
    for a in [0,2,4,6,8,10,12,13,14,16,18]:
        t2to19policy = utils.consecutive_stage_policy(utils.DeterministicPolicy(a), horizon-2)
        t0to1policy = [utils.DeterministicPolicy(1),utils.DeterministicPolicy(a)]
        full_policy = utils.concat_long_policy(t0to1policy,t2to19policy)
        sample_policy_list += [full_policy]

    # s1 = 1
    for a in range(horizon):        
        t2to19policy = utils.consecutive_stage_policy(utils.DeterministicPolicy(a), horizon-2)
        t0to1policy = [utils.DeterministicPolicy(13),utils.DeterministicPolicy(a)]
        full_policy = utils.concat_long_policy(t0to1policy,t2to19policy)
        sample_policy_list += [full_policy]
    
    return sample_policy_list

def env_linearq_policy(base_0_num):
    '''
    input: used to define the optimal policy
    '''
    sample_policy_list = []
    optimal_policy = full_policies.bad_good_policy(base_0_num)
    sample_policy_list.append(optimal_policy)
    for h in range(horizon):
        new_policy = full_policies.bad_good_policy(base_0_num)
        new_policy[h] = new_policy[h].inverse_prob()
        sample_policy_list.append(new_policy)

    return sample_policy_list



# sample_policy_list = [all1_policy, all0_policy, alt01_policy, alt10_policy]
# sample_policy_list = [alt01_policy, alt10_policy, all1_policy, all0_policy]
# sample_policy_list = [full_policies.repeat_action_policy(i) for i in range(num_actions)]
# sample_policy_list = [full_policies.double_loop_full_policy(i,j,num_actions) for j in range(num_actions) for i in range(num_actions)]
# sample_policy_list = env_hard_policy(horizon)
# sample_policy_list = [full_policies.loop_full_policy(i, num_actions) for i in range(num_actions)]
sample_policy_list = env_linearq_policy(10)

# num_repeat_list = [1 for _ in range(num_actions)]
num_repeat_list = [1 for _ in sample_policy_list]

num_repeat_list[0] = horizon

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

        

