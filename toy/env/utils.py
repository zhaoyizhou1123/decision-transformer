import torch
import torch.nn.functional as f

def read_env(env_path):
    '''
    Input: env_path, string, path to environment description file. \n
    Output: 
    - state_space: list of states, each state Tensor with shape (state_dim=1)
    - action_space: list of actions, each action Tensor with shape (action_dim=1)
    - horizon: int, horizon of the game
    - init_states: Tensor (num_states), initial state distribution
    - P: dict, key is tuple (s,a), s,a both int; value is state distribution, Tensor with shape (num_states)
    - r: dict, key is tuple (s,a), s,a both int; value is reward r(s,a), scalar
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

        # state, action space
        state_space = [torch.tensor([i]) for i in range(num_states)]
        action_space = [torch.tensor([i]) for i in range(num_actions)]

        # horizon
        horizon = int(lines[2])

        # initial_states
        init_states_split = lines[3].split(",")
        init_states = [float(prob) for prob in init_states_split]
        init_states = torch.tensor(init_states)

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
            P[(s,a)] = probs # add a key:value pair in P

        # Create r
        reward_lines = lines[-num_states*num_actions:]
        r = {}
        for line in reward_lines:
            split_line = line.split(",") # split the line into s,a,prob0,prob1,...
            s = int(split_line[0])
            a = int(split_line[1])
            reward = float(split_line[2])
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
    0 -> (1,0), 1 -> (0,1)
    x: torch.Tensor (1) or (n) (inv True) or torch.Tensor(n,2), (2) (inv False)
    output: (2), torch.Tensor(n,2) (inv True) or torch.Tensor(n), (1) (inv False)
    '''
    if not inv:
        enc = f.one_hot(x.long(), 2) # (1,2), (n,2), convert to int64 to avoid bugs
        return enc.squeeze(dim = 0) #(2), (n,2)
    else:
        x = x.reshape(-1, 2) #(n,2), (1,2)
        return torch.argmax(x, dim=1) 


# state_space, action_space, horizon, init_states, P, r = read_env("env_rev.txt")
# for key, value in P:
#     print(f"key:{type(key.item())}")
#     print(f"value:{value}")

# dict = {(1,1):2, (2,2):3}
# for key, value in dict:
#     print(f"key:{key}")
#     print(f"value:{value}")

# print(dict[(1,1)])

# a = torch.tensor([int(0)])
# P[(a,a)]
# print(type(a))