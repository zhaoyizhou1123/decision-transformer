import torch
import torch.nn.functional as f

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
        num_states = int(lines[0]) # '\n' is omitted in int()
        num_actions = int(lines[1])

        # horizon
        horizon = int(lines[2])

        # state, action space
        if not time_depend_s:
            state_space = [torch.tensor([i]) for i in range(num_states)]
        else:
            state_space = [] # will be 0,1,...S-1,S,S+1,...,HS-1
            for t in range(horizon):
                for s in range(num_states):
                    state_space.append(torch.tensor([s+t*num_states]))

        if not time_depend_a:
            action_space = [torch.tensor([i]) for i in range(num_actions)]
        else: 
            action_space = [] # will be 0,1,...A-1,...,HA-1
            for t in range(horizon):
                for a in range(num_actions):
                    action_space.append(torch.tensor([a+t*num_actions]))

        # initial_states
        init_states_split = lines[3].split(",")
        init_states = [float(prob) for prob in init_states_split]
        assert len(init_states) == num_states, f"Length of init_states mismatch with state space"
        init_states = torch.tensor(init_states)
        

        if time_depend_s:
            init_states = torch.cat([init_states, torch.zeros((horizon-1)*num_states)], dim=0) # (HS,)

        # Create P
        transition_lines = lines[-2*num_states*num_actions:-num_states*num_actions] # Count backwards
        P = {}
        for line in transition_lines:
            split_line = line.split(",") # split the line into s,a,prob0,prob1,...
            s = int(split_line[0])
            a = int(split_line[1])
            probs = split_line[2:]
            probs = [float(prob) for prob in probs]
            probs = torch.tensor(probs) # Tensor of floats

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
        reward_lines = lines[-num_states*num_actions:]
        r = {}
        for line in reward_lines:
            split_line = line.split(",") # split the line into s,a,prob0,prob1,...
            s = int(split_line[0])
            a = int(split_line[1])
            reward = float(split_line[2])

            s_loop = horizon if time_depend_s else 1
            a_loop = horizon if time_depend_a else 1
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

# state_space, action_space, horizon, init_states, P, r = read_env("env_bandit.txt", time_depend=True)
# print(len(state_space), len(action_space))
# print(P)
# for key, value in P:
#     print(f"key:{key}")
#     print(f"value:{value}")

# dict = {(1,1):2, (2,2):3}
# for key, value in dict:
#     print(f"key:{key}")
#     print(f"value:{value}")

# print(dict[(1,1)])

# a = torch.tensor([int(0)])
# P[(a,a)]
# print(type(a))