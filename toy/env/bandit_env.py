from env.utils import read_env, sample as default_sample
import torch

class BanditEnv:
    '''
    A repeat bandit env of horizon H. \n
    Modification: timestep starts with 0
    '''
    # metadata = {'render.modes': ['human']}
    def __init__(self, env_path, sample=None, state_hash=None, action_hash=None, 
                 time_depend_s=False, time_depend_a=False):
        '''
        - env_path: str, path to environment description file
        - sample: function. A way to sample int from given probs. Should be utils.sample
        - state_hash: function | None. If not None, specifies a way to hash states to observations
        - action_hash: function | None. If not None, specifies a way to hash actions. Syntax:
            `action_hash(action, inv=False)`. 
        - time_depend_s: bool. If true, state becomes s+t*S 
        - time_depend_a: bool. If true, action becomes a+t*A

        Note: the class stores true states, specified by env description file. It outputs obs, which 
        is hashed from state by state_hash. 
        '''
        # super(BanditEnv, self).__init__()

        # self.df = df
        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        # print(time_depend)
        state_space, action_space, horizon, init_states, P, r = read_env(env_path, time_depend_s, time_depend_a)
        # print(state_space, action_space, horizon, init_states, P, r)

        self.state_space = state_space # list of states, each state Tensor with shape (1)
        self.action_space = action_space # list of actions, each action Tensor with shape (1)
        self.init_state_probs = init_states # Tensor (num_states)
        self._HORIZON = horizon # int
        self.transition_table = P # dict, (s,a):probs
        self.reward_table = r # dict, (s,a):r
        
        self._timestep = 0 # current timestep, starting with 0
        
        self._return = 0 # total rewards
        self._state_hash = state_hash # None or int to int function
        self._action_hash = action_hash # None or int to int function

        if sample is not None:
            self._basic_sample = sample # No hashing sample function
        else:
            self._basic_sample = default_sample

        self._current_state = self._basic_sample(self.init_state_probs) # True state
        # self._current_state = self._hash_state(current_state)

    def step(self, observed_action):
        '''
        observed_action: Tensor with shape [1], or [[[[...[1]]]]], or int
        Return: next_obs: next observation; reward: equals action; done: bool, True if H actions taken 
        '''
        # Decode action 
        action = self._hash_action(observed_action, inv = True)
        # Change action to int
        if hasattr(action, 'item'): # action is Tensor type, change to scalr
            action = action.item()
        action = int(action)
        current_state = self._current_state.item() # int

        next_state_probs = self.transition_table[(current_state, action)]
        next_state = self._basic_sample(next_state_probs)
        # next_state= self._hash_state(next_state) # hash the state if needed
        self._current_state = next_state # update state
        
        reward = self.reward_table[(current_state, action)]
        self._return += reward

        self._timestep += 1
        done = (self._timestep == self._HORIZON)
              
        return self._hash_state(self._current_state), reward, done
    
    def reset(self):
        '''Return initial state as torch.Tensor of shape (1,)'''
        current_state = self._basic_sample(self.init_state_probs)
        self._current_state = self._hash_state(current_state)

        self._timestep = 0
        self._return = 0
        return self._current_state
        
    def render(self):
        print(f"Current observation is {self._hash_state(self._current_state)}")
        print(f"Time step={self._timestep}")
        print(f"Current total reward is {self._return}")

    def get_state(self):
        '''Return current observation as Tensor(state_dim)'''
        return self._hash_state(self._current_state)
    
    def get_action(self, metrics, mode='best'):
        '''
        Sample action according to metrics given. Testing should use this method to get action\n
        Input: 
        - metrics, Tensor of (num_action)
        - mode: 'best' or 'sample'. If best, get the action with highest metric. If sample, 
        Treat metrics as probs.
        Output: an action observation
        '''
        assert len(metrics.shape) ==1 and metrics.shape[0] == self.get_num_action(), f"Invalide metrics shape {metrics.shape}"
        assert mode == 'best' or mode == 'sample', f"Invalid mode {mode}"
        if mode == 'best':
            opt_index = torch.argmax(metrics).item()
            action = torch.tensor([opt_index])
        else: # sample mode
            assert torch.sum(metrics) == 1, f"Sample mode, but metrics sum is not 1: {metrics}"
            action = self._basic_sample(metrics)
        
        return self._hash_action(action)

    def get_num_action(self):
        '''
        Return the number of possible actions. For initializing empty action Tensor
        '''
        return len(self.action_space)
    
    def get_action_space(self):
        '''
        Output: hashed action space, Tensor(num_actions, action_dim)
        '''
        list_action_space = []
        for action in self.action_space:
            hashed_action = self._hash_action(action).reshape(-1) # Tensor(hashed_dim), remove extra dimension in one_hot_hash
            if hashed_action.shape[0] == 1: # 1-dim
                hashed_action = [hashed_action.item()]
            else: # >2 dim
                hashed_action = hashed_action.tolist()
            list_action_space += [hashed_action]
        # print(f"list_action_space {list_action_space}")
        return torch.tensor(list_action_space).type(torch.float32)
    
    def get_horizon(self):
        return self._HORIZON
    
    def _hash_state(self, state):
        '''
        Return the hashed state according to self._state_hash \n
        Input: state, Tensor (1)
        Output: state, Tensor (1)
        '''
        if self._state_hash is not None:
            hashed_state = self._state_hash(state)
            return hashed_state
        else:
            return state
        
    def _hash_action(self, action, inv=False):
        '''
        Return the hashed action according to self._action_hash \n
        Input: 
        - Action is the true/observed action. Tensor(n) or scalar
        - inv=False implements action -> obs; inv=True implements obs -> action. 

        Output: Same type as input
        '''
        if self._action_hash is not None:
            hashed_action = self._action_hash(action, inv)
            return hashed_action
        else:
            return action
        