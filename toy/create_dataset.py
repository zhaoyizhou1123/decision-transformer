# Sample a dataset, store in a csv file

import numpy.random as random
from env.no_best_RTG import BanditEnv as Env
# from env.no_best_RTG import BanditEnvReverse as Env
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_file', type=str, default='./dataset/toy_alternate.csv')
parser.add_argument('--horizon', type=int, default=20)
parser.add_argument('--num_trajectories', type=int, default=10)
args = parser.parse_args()

class Policy:
    '''Abstraction of a step policy for the toy environment'''
    def __init__(self, prob0):
        assert 0<=prob0 and prob0<=1, "Invalid prob0!"
        self._prob0 = prob0     # _prob0: probability of taking action 0
        self._prob1 = 1 - prob0 # _prob1: probability of taking action 1
    
    def sample_action(self):
        '''
        Sample an action according to the policy
        '''
        action = random.choice(a=[0,1], size=1, p=[self._prob0, self._prob1]) # an array of size 1
        return action[0]

# Define two stage policy instances
good_policy = Policy(prob0 = 0)
bad_policy = Policy(prob0 = 1)

# Create Env instance
env = Env(args.horizon)

with open(args.output_file, "w") as f:
    # Write annotation
    for i in range(1, args.horizon+1):
        f.write(f"t{i},s{i},a{i},r{i},")
    f.write(f"t{args.horizon+1}\n")
    # Sample. 
    for epoch in range(args.num_trajectories//2):
        # 5 steps good, then all bad
        # for h in range(1,6):
        #     state = env.get_state()
        #     f.write(f"{h},{state},")
        #     # Always run the best policy
        #     action = good_policy.sample_action()
        #     _, reward, _ = env.step(action)
        #     f.write(f"{action},{reward},")    
        # for h in range(6, args.horizon+1):
        #     state = env.get_state()
        #     f.write(f"{h},{state},")
        #     # Always run the worst policy
        #     action = bad_policy.sample_action()
        #     _, reward, _ = env.step(action)
        #     f.write(f"{action},{reward},")
        # f.write(f"{args.horizon+1}\n")

        # good,bad,good,...
        for h in range(20):
            state = env.get_state()
            f.write(f"{h},{state},")
            if h % 2 == 0:
                action = good_policy.sample_action()
            else:
                action = bad_policy.sample_action()
            _, reward, _ = env.step(action)
            f.write(f"{action},{reward},")
        f.write(f"{args.horizon+1}\n")
    for epoch in range(args.num_trajectories//2, args.num_trajectories):
        # # 5 steps bad, then all good
        # for h in range(1,6):
        #     state = env.get_state()
        #     f.write(f"{h},{state},")
        #     # Always run the best policy
        #     action = bad_policy.sample_action()
        #     _, reward, _ = env.step(action)
        #     f.write(f"{action},{reward},")    
        # for h in range(6, args.horizon+1):
        #     state = env.get_state()
        #     f.write(f"{h},{state},")
        #     # Always run the worst policy
        #     action = good_policy.sample_action()
        #     _, reward, _ = env.step(action)
        #     f.write(f"{action},{reward},")
        # f.write(f"{args.horizon+1}\n")

        # bad,good,bad,...
        for h in range(20):
            state = env.get_state()
            f.write(f"{h},{state},")
            if h % 2 == 1:
                action = good_policy.sample_action()
            else:
                action = bad_policy.sample_action()
            _, reward, _ = env.step(action)
            f.write(f"{action},{reward},")
        f.write(f"{args.horizon+1}\n")

        

