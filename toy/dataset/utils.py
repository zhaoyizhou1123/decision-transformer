import numpy.random as random

class Policy:
    def __init__(self) -> None:
        pass
    def sample_action(self):
        return None
    

class Policy01(Policy):
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
    
class DeterministicPolicy(Policy):
    def __init__(self, action):
        self._action = action
    def sample_action(self):
        return self._action

# Define two stage policy instances
good_policy = Policy01(prob0 = 0) # Always 1, may not be really good
bad_policy = Policy01(prob0 = 1) # Always 0, may not be really bad
rand_policy = Policy01(prob0=0.5)

# Represent policies in list[Policy]
# Some helper functions to create long policies
def consecutive_stage_policy(stage_policy: Policy, steps: int):
    '''
    Return a list of [stage_policy] for length [steps].
    stage_policy: Policy
    steps: int
    '''
    return [stage_policy for i in range(steps)]
def alt_stage_policy(policy1: Policy, policy2: Policy, steps: int):
    '''
    Alternating policies: policy1, policy2, policy1,... Length steps
    '''
    long_policy = []
    for i in range(steps):
        if i % 2 == 0:
            long_policy.append(policy1)
        else:
            long_policy.append(policy2)
    return long_policy

def concat_long_policy(long_policy1: list, long_policy2: list):
    '''
    Return concatanated list of long_policy1 and long_policy2
    '''
    return long_policy1 + long_policy2

# def good_bad_seq_policy(self, length_list: list(int), start_good: bool):
#     '''
#     Return [good_slice, bad_slice, good_slice, ...]. 
#     - length_list: specify the length of each slice
#     - start_good: starting with good policy if true, else bad
#     '''

class FullPolicy:
    '''
    Define some common policies for the whole horizon. \n
    All policies are given in list of Policy
    '''
    def __init__(self, horizon: int):
        self.horizon = horizon

    def all_good_policy(self):
        return consecutive_stage_policy(good_policy, self.horizon)
    def all_bad_policy(self):
        return consecutive_stage_policy(bad_policy, self.horizon)
    def all_rand_policy(self):
        return consecutive_stage_policy(rand_policy, self.horizon)
    def good_bad_policy(self, good_len: int):
        '''
        Good policy for [good_len] times, then bad
        '''
        good_sice = consecutive_stage_policy(good_policy, good_len)
        bad_slice = consecutive_stage_policy(bad_policy, self.horizon - good_len)
        return concat_long_policy(good_sice, bad_slice)
    def bad_good_policy(self, bad_len: int):
        '''
        Bad policy for [bad_len] times, then good
        '''
        bad_slice = consecutive_stage_policy(bad_policy, bad_len)
        good_slice = consecutive_stage_policy(good_policy, self.horizon - bad_len)
        return concat_long_policy(bad_slice, good_slice)
    def alt_good_bad_policy(self):
        '''
        Alternating good,bad,good,...
        '''
        return alt_stage_policy(good_policy, bad_policy, self.horizon)
    def alt_bad_good_policy(self):
        '''
        Alternating good,bad,good,...
        '''
        return alt_stage_policy(bad_policy, good_policy, self.horizon)
    
    def repeat_action_policy(self, action):
        '''
        Repeatedly choose action
        '''
        stage_policy = DeterministicPolicy(action)
        return consecutive_stage_policy(stage_policy, self.horizon)
    
    def double_loop_full_policy(self, start_action1, start_action2, num_action):
        '''
        horizon must be even
        action form a,a+1,...,a+H/2-1 % A
        '''
        assert self.horizon % 2 == 0, f"Horizon {self.horizon} is not an even number!"
        loop = self.horizon // 2
        full_policy = []
        for t in range(loop):
            full_policy.append(DeterministicPolicy((start_action1 + t) % num_action))
            full_policy.append(DeterministicPolicy((start_action2 + t) % num_action))
        return full_policy
    

