from typing import Any
import torch
import torch.nn.functional as f
import math

def read_env(env_path, time_depend_s=False, time_depend_a=False):
    '''
    Input: env_path, string, path to environment description file. \n
    Output: 
    - state_space: list of states, each state Tensor with shape (state_dim=1)
    - action_space: list of actions, each action Tensor with shape (action_dim=1)
    - horizon: int, horizon of the game
    - init_states: Tensor (num_states), initial state distribution
    - P: dict, key is tuple (s,a), s,a both int; value is state distribution, Tensor with shape (num_states)
    - r: dict, key is tuple (s,a), s,a both int; value is reward r(s,a), scalar
    - time_depend_s: bool. If true, state becomes s+t*S 
    - time_depend_a: bool. If true, action becomes a+t*A
        Then P has H^2*SA keys, value has shape (HS). r has H^2*SA keys.
    (Note: dict with tensor as key is problematic, as two tensors of same value can be different objects)
    '''
    with open(env_path, "r") as f:
        all_lines = f.readlines()
        
        # Remove comments and empty lines
        lines = []
        for line in all_lines:
            if line[0] != "\n" and line[0] != "#":
                lines.append(line)

        num_states = int(lines[0])

        # Num states
        # state_line = lines[0].split(";")
        # num_equiv_states = int(state_line[0]) # '\n' is omitted in int(), effective state num
        # Construct state mapping
        # if len(state_line) == 1: # Default state mapping
        #     state_map = {s:[s] for s in range(num_equiv_states)} # map effective state to list of true states
        # else:
        #     assert num_equiv_states == len(state_line) - 1, f"Number of state mapping is not 1 or num_equiv_states!"
        #     state_map = {}
        #     for s in range(num_equiv_states):
        #         true_state_str_list = state_line[s+1].split(",")
        #         true_state_list = [int(state) for state in true_state_str_list]
        #         state_map[s] = true_state_list

        # Num actions
        action_line = lines[1].split(";")
        num_equiv_actions = int(action_line[0]) # effective action num
        # Construct action mapping
        if len(action_line) == 1: # Default action mapping
            action_map = {a:[a] for a in range(num_equiv_actions)} # map effective action to list of true actions
        else:
            assert num_equiv_actions == len(action_line) - 1, f"Number of action mapping is not 1 or num_equiv_actions!"
            action_map = {}
            for s in range(num_equiv_actions):
                true_action_str_list = action_line[s+1].split(",")
                true_action_list = [int(action) for action in true_action_str_list]
                action_map[s] = true_action_list

        # horizon
        horizon = int(lines[2])

        # state space
        full_state_list = list(range(num_states))
        # full_state_list = []
        # for _, state_list in state_map.items():
        #     full_state_list += state_list
        # full_state_list.sort() # sort the states
        # num_states = len(full_state_list)

        if not time_depend_s:
            state_space = [torch.tensor([i]) for i in full_state_list]
        else:
            state_space = [] # will be 0,1,...S-1,S,S+1,...,HS-1
            for t in range(horizon):
                for s in full_state_list:
                    state_space.append(torch.tensor([s+t*num_states]))

        # action space
        full_action_list = []
        for _, action_list in action_map.items():
            full_action_list += action_list
        full_action_list.sort() # sort actions
        num_actions = len(full_action_list)

        if not time_depend_a:
            action_space = [torch.tensor([i]) for i in full_action_list]
        else: 
            action_space = [] # will be 0,1,...A-1,...,HA-1
            for t in range(horizon):
                for a in full_action_list:
                    action_space.append(torch.tensor([a+t*num_actions]))

        # initial_states
        init_states_split = lines[3].split(",")
        init_states = [float(prob) for prob in init_states_split]
        assert len(init_states) == num_states, f"Length of init_states mismatch with state space"
        init_states = torch.tensor(init_states)
        

        if time_depend_s:
            init_states = torch.cat([init_states, torch.zeros((horizon-1)*num_states)], dim=0) # (HS,)

        # Create P
        transition_lines = lines[-2*num_states*num_equiv_actions:-num_states*num_equiv_actions] # Count backwards
        P = {}
        for line in transition_lines:
            split_line = line.split(",") # split the line into s,a,prob0,prob1,...
            s = int(split_line[0])
            equiv_a = int(split_line[1])
            probs = split_line[2:]
            probs = [float(prob) for prob in probs]
            probs = torch.tensor(probs) # Tensor of floats

            for a in action_map[equiv_a]:
                if not time_depend_s:
                    if not time_depend_a:
                        P[(s,a)] = probs # add a key:value pair in P
                    else: # time_depend_a = True
                        for h in range(horizon):
                            P[(s, a+h*num_actions)] = probs 
                else: # time_depend_s = True
                    for t in range(horizon): # loop for s variants on t
                        whole_probs = torch.zeros((horizon)*num_states)
                        whole_probs[(t+1)*num_states : (t+2)*num_states] = probs
                        if t == horizon-1: # Terminal state, t keeps same
                            whole_probs[t*num_states : (t+1)*num_states] = probs
                        # Ideally, s,a should belong to same t. But in case the algo misbehaves, we follow the time of state.
                        if not time_depend_a:
                            P[(s+t*num_states, a)] = whole_probs
                        else:
                            for h in range(horizon):
                                P[(s+t*num_states, a+h*num_actions)] = whole_probs 

        # Create r
        reward_lines = lines[-num_states*num_equiv_actions:]
        r = {}
        for line in reward_lines:
            split_line = line.split(",") # split the line into s,a,prob0,prob1,...
            s = int(split_line[0])
            equiv_a = int(split_line[1])
            reward = float(split_line[2])

            s_loop = horizon if time_depend_s else 1
            a_loop = horizon if time_depend_a else 1
            for a in action_map[equiv_a]:
                for t in range(s_loop):
                    for h in range(a_loop):
                        r[(s+t*num_states, a+h*num_actions)] = reward
            # if not time_depend:
            #     r[(s,a)] = reward
            # else:
            #     for t in range(horizon):
            #         for h in range(horizon):
            #             r[(s+t*num_states, a+h*num_actions)] = reward

    return state_space, action_space, horizon, init_states, P, r

def read_env_linearq(env_path: str):
    '''
    For linear Q environments. \n
    Input: env_path, string, path to environment description file. \n
    Output: 
    - state_space: list of states, each state Tensor with shape (state_dim=1)
    - action_space: list of actions, each action Tensor with shape (action_dim=1)
    - horizon: int, horizon of the game
    - init_states: Tensor (num_states), initial state distribution
    - P: dict, key is tuple (s,a), s,a both int; value is state distribution, Tensor with shape (num_states)
    - r: dict, key is tuple (s,a), s,a both int; value is reward r(s,a), scalar
    '''
    with open(env_path, "r") as f:
        all_lines = f.readlines()
        
        # Remove comments and empty lines
        lines = []
        for line in all_lines:
            if line[0] != "\n" and line[0] != "#":
                lines.append(line)

    # Num actions
    num_actions = int(lines[0])
    action_space = [torch.tensor(a) for a in range(num_actions)]

    # Q-functions, given as a class
    qf_lines = lines[-num_actions:]
    q_functions = [] # list of Q-functions, index by a
    s_intercepts = [] # s_intercepts

    class LinearQ:
        def __init__(self, point_s, point_q, slope):
            self.point_s = point_s
            self.point_q = point_q
            self.slope = slope
        def __call__(self, s):
            return max(0, self.slope * (s - self.point_s) + self.point_q)

    for qf_line in qf_lines: # one action
        split_line = qf_line.split(',')
        point_s = float(split_line[0])
        point_q = float(split_line[1])
        slope = float(split_line[2])
        assert slope < 0, f"Slope should be negative, get {slope}"
        q_functions.append(LinearQ(point_s, point_q, slope))
        s_intercepts.append(point_s - point_q / slope)

    # print(q_functions)

    # Deduce state space
    max_state = math.ceil(max(s_intercepts))
    state_space = [torch.tensor(s) for s in range(max_state + 1)]
    num_states = max_state + 1

    # Get horizon
    horizon = max_state + 1

    # Get init_states, always at 0
    init_states = torch.zeros(num_states)
    init_states[0] = 1

    # Get tabular Q-functions, value functions, and optimal actions
    # Tabular Q: list(list), state, action
    # value: list, index state
    # opt_act_table: list, index state
    q_table = []
    v_table = []
    # opt_act_table = []
    for s in range(num_states):
        qfs = [(q_functions[a])(s) for a in range(num_actions)]
        q_table.append(qfs)

        value = max(qfs)
        # opt_act = qfs.index(value) # optimal action

        v_table.append(value)
        # opt_act_table.append(opt_act)

    # print(q_table)
    # print(v_table)

    # Deduce transition
    P = {}
    for s in range(num_states):
        if s == num_states - 1: # The last state, transits to itself
            probs = torch.zeros(num_states)
            probs[num_states - 1] = 1
            for a in range(num_actions):
                P[(s,a)] = probs
        else: # not the last state
            v = v_table[s] # value function
            for a in range(num_actions):
                qf = q_table[s][a] # q function of a
                if qf == v: # optimal action
                    # transits to the next state
                    probs = torch.zeros(num_states)
                    probs[s+1] = 1
                    P[(s,a)] = probs
                else: # sub-optimal action
                    # use an odd state as reference state
                    ref_s = s if s % 2 == 1 else s + 1 
                    qf_ref_s = q_table[ref_s][a]
                    
                    # If is 0, transits directly to the last state
                    if qf_ref_s == 0:
                        next_s = num_states -1
                    # If still > 0, find the largest value function below qf_ref_s
                    else:
                        for next_s in range(ref_s, num_states):
                            if v_table[next_s] < qf_ref_s:
                                break
                
                    probs = torch.zeros(num_states)
                    probs[next_s] = 1
                    P[(s,a)] = probs

    # Deduce reward
    r = {}
    for s in range(num_states):
        for a in range(num_actions):
            next_s_probs = P[(s,a)]
            next_s = torch.argmax(next_s_probs, dim=0).item()
            reward = q_table[s][a] - v_table[next_s]
            r[(s,a)] = reward

    return state_space, action_space, horizon, init_states, P, r    

        
        


def sample(probs):
    '''
    Sample according to probs given. \n
    Input: probs, Tensor of (n)
    Output: Tensor[int], int in [0,n-1]
    '''
    return torch.multinomial(probs, num_samples=1)

def reversible_hash(x, inv=False):
    '''
    f(x)=100x-50. 
    x can be state/action ...
    Input: x, scalar or np.array or Tensor(n). inv: bool, present inverse computation if True
    Output: same type as Input
    '''
    if not inv:
        return 100*x-50
    else:
        return 0.01*(x+50)
    
def one_hot_hash(x, inv=False):
    '''
    0 -> (1,0), 1 -> (0,1) \n
    - x: torch.Tensor (1) or (n) (inv False) or torch.Tensor(n,2), (2) (inv True)
    - 2: number of inputs
    output: (2), torch.Tensor(n,2) (inv False) or torch.Tensor(n), (1) (inv True)
    '''
    if not inv:
        enc = f.one_hot(x.long(), 2) # (1,2), (n,2), convert to int64 to avoid bugs
        return enc.squeeze(dim = 0) #(2), (n,2)
    else:
        x = x.reshape(-1,x.shape[-1]) #(n,2), (1,2)
        return torch.argmax(x, dim=1) 
    
class OneHotHash:
    '''
    one_hot_hash with designated num_class
    '''
    def __init__(self, num_class) -> None:
        self.num_class = num_class
    def hash(self, x, inv=False):
        if not inv:
            enc = f.one_hot(x.long(), self.num_class) # (1,self.num_class), (n,self.num_class), convert to int64 to avoid bugs
            return enc.squeeze(dim = 0) #(self.num_class), (n,self.num_class)
        else:
            x = x.reshape(-1,x.shape[-1]) #(n,self.num_class), (1,self.num_class)
            return torch.argmax(x, dim=1) 

# state_space, action_space, horizon, init_states, P, r = read_env_linearq("env_linearq.txt")
# print(len(state_space), len(action_space))
# for s in range(20):
#     for a in range(2):
#         next_s_probs = P[(s,a)]
#         next_s = torch.argmax(next_s_probs).item()
#         print(f"s={s},a={a},s'={next_s}")
# print('---------------------')
# print(r)

# dict = {(1,1):2, (2,2):3}
# for key, value in dict:
#     print(f"key:{key}")
#     print(f"value:{value}")

# print(dict[(1,1)])

# a = torch.tensor([int(0)])
# P[(a,a)]
# print(type(a))