from env.utils import sample as default_sample
import torch

class TimeVarEnv:
    '''
    A specific time-variant environment. \n
    State: 0->1/2/3->0....  \n
    Horizon: horizon=H even \n
    Action: num_actions=K (typically H/2) possible actions. \n
    Transition: each round (0->1/2/3->0) has an a*. Typically, a* = [h/2] From 0,
    a* gets to state 3, other even gets to 2, other odd gets to 1. State 1,2,3 always transits to 0. \n
    Reward: State 0: a* and other odd gets 1, other even gets 2. At state 1, other odd gets 2, rest 0;
    state 2, other even gets 1, rest 0; state 3, a* gets 3, other 0. \n
    Optimal action: Always choose the a* of the timestep. Return: 2H. \n

    Dataset
    ==
    timevar_exp: double_loop_full(i,j) i,j in 0,1,...,9. Problem: Optimal policy too rare
    timevar_exp2: Optimal policy repeated 40 times
    '''
    def __init__(self, horizon: int, num_actions: int, state_hash=None, action_hash=None):
        assert horizon % 2 == 0, f"Horizon must be even!"
        self.horizon = horizon
        self.num_actions = num_actions
        self.num_states = int(4)

        self._state_hash = state_hash # None or int to int function
        self._action_hash = action_hash # None or int to int function

        self._basic_sample = default_sample

        self.state_space = [torch.tensor([s]) for s in range(self.num_states)]
        self.action_space = [torch.tensor([a]) for a in range(self.num_actions)]

        # initial state
        self._initial_state = torch.tensor([0])

        

        self.reset()
        
        # To define the transition and reward
        self.opt_action_table = {h: (h // 2) % self.num_actions for h in range(self.horizon)} # dict[int, int]

    def _get_next_state(self, state, action, timestep):
        '''
        Transition.
        - state: Tensor (1,)
        - action: Tensor(1,)
        - timestep: scalar
        Return: Tensor (1,)
        '''
        state = state.item()
        action = action.item()
        opt_action = self.opt_action_table[timestep]
        assert state in range(self.num_states), f"Transition: Invalid state {state}!"
        if state == 0:
            if action == opt_action:
                next_state = int(3)  
            elif action % 2 == 1:
                next_state = int(1)
            else:
                next_state = int(2)
        else:
            next_state = int(0)
        return torch.tensor([next_state])
    
    def _get_reward(self, state, action, timestep):
        '''
        Reward
        '''
        
        # Choose an optimal action for each step

        state = state.item()
        action = action.item()

        assert state in range(self.num_states), f"Reward: Invalid state {state}"
        assert action in list(range(self.num_actions)), f"Reward: Invalid action {action}"
        assert 0 <= timestep and timestep < self.horizon, f"Reward: Invalid timestep {timestep}"

        opt_action = self.opt_action_table[timestep]

        # if action == opt_action:
        #     reward = 0 if state == 0 else 3
        # elif action % 2 == 0:
        #     reward = 2 if state == 0 else 0
        # else: # odd and not optimal
        #     reward = 0 if state == 0 else 2
        if state == 0:
            if action == opt_action:
                reward = 1
            elif action % 2 == 0:
                reward = 2
            else:
                reward = 1
        elif state == 1:
            reward = 2 if action % 2 == 1 else 0

            # if action == opt_action:
            #     reward = 0
            # elif action % 2 == 0:
            #     reward = 1
            # else:
            #     reward = 2
        elif state == 2:
            reward = 1 if action % 2 == 0 else 0
            # if action == opt_action:
            #     reward = 3
            # else:
            #     reward = 0
        else:
            reward = 3 if action == opt_action else 0
        
        
        return reward


    def step(self, observed_action):
        '''
        observed_action: encoded action
        '''
        action = self._hash_action(observed_action, inv = True)
        if not hasattr(action, 'item'): # action is Tensor type, change to scalr
            action = torch.tensor(action)
        action = action.reshape(-1)

        next_state = self._get_next_state(self._current_state, action, self._timestep)
        reward = self._get_reward(self._current_state, action, self._timestep)

        # Update environment info
        self._current_state = next_state
        self._return += reward
        self._timestep += 1

        done = (self._timestep == self.horizon)

        return next_state, reward, done
    
    def reset(self):
        '''Return initial state as torch.Tensor of shape (1,)'''
        self._current_state = self._initial_state

        self._timestep = 0
        self._return = 0

        return self._current_state
    
    def get_state(self):
        return self._current_state
    
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
        return self.horizon
    
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

class TimeVarBanditEnv:
    '''
    A specific time-variant environment. \n
    State: 0->0....  \n
    Horizon: horizon=H \n
    Action: num_actions=K (typically H) possible actions. \n 
        Each h step has an a*(h), typically a*(h)=h%K
    Reward: step h: a*(h) gets 1, other 0
    Optimal action: Always choose the a* of the timestep. Return: H. \n

    Dataset
    ==
    timevarbandit_exp: a,a+1,...,a+H-1 (a=0,1,...H-1)
    '''
    def __init__(self, horizon: int, num_actions: int, state_hash=None, action_hash=None):
        self.horizon = horizon
        self.num_actions = num_actions
        self.num_states = int(1)

        if state_hash is not None or action_hash is not None:
            print(f"Warning: state_hash or action_hash is not completely implemented for Timevar envs!")

        self._state_hash = state_hash # None or int to int function
        self._action_hash = action_hash # None or int to int function

        self._basic_sample = default_sample

        self.state_space = [torch.tensor([s]) for s in range(self.num_states)]
        self.action_space = [torch.tensor([a]) for a in range(self.num_actions)]

        # initial state
        self._initial_state = torch.tensor([0])

        self.reset()
        
        # To define the transition and reward
        self.opt_action_table = {h: h % self.num_actions for h in range(self.horizon)} # dict[int, int]

    def _get_next_state(self, state, action, timestep):
        '''
        Transition.
        - state: Tensor (1,)
        - action: Tensor(1,)
        - timestep: scalar
        Return: Tensor (1,)
        '''
        return torch.tensor([0])
    
    def _get_reward(self, state, action, timestep):
        '''
        Reward
        '''
        
        # Choose an optimal action for each step

        state = state.item()
        action = action.item()

        assert state in range(self.num_states), f"Reward: Invalid state {state}"
        assert action in list(range(self.num_actions)), f"Reward: Invalid action {action}"
        assert 0 <= timestep and timestep < self.horizon, f"Reward: Invalid timestep {timestep}"

        opt_action = self.opt_action_table[timestep]

        # if action == opt_action:
        #     reward = 0 if state == 0 else 3
        # elif action % 2 == 0:
        #     reward = 2 if state == 0 else 0
        # else: # odd and not optimal
        #     reward = 0 if state == 0 else 2      
        
        return 1 if action == opt_action else 0


    def step(self, observed_action):
        '''
        observed_action: encoded action
        '''
        action = self._hash_action(observed_action, inv = True)
        if not hasattr(action, 'item'): # action is Tensor type, change to scalr
            action = torch.tensor(action)
        action = action.reshape(-1)

        next_state = self._get_next_state(self._current_state, action, self._timestep)
        reward = self._get_reward(self._current_state, action, self._timestep)

        # Update environment info
        self._current_state = next_state
        self._return += reward
        self._timestep += 1

        done = (self._timestep == self.horizon)

        return next_state, reward, done
    
    def reset(self):
        '''Return initial state as torch.Tensor of shape (1,)'''
        self._current_state = self._initial_state

        self._timestep = 0
        self._return = 0

        return self._current_state
    
    def get_state(self):
        return self._current_state
    
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
        return self.horizon
    
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