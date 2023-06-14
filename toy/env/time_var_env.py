from env.utils import sample as default_sample
import torch

class TimeVarEnv:
    '''
    A specific time-variant environment. \n
    State: 0->1->0....
    Action: num_actions=K possible actions.
    Horizon: horizon=H
    reward: each round (0->1->0) has an a*. a* gets reward 0,3; Others 2,0 or 0,2.
    '''
    def __init__(self, horizon: int, num_actions: int, state_hash=None, action_hash=None):
        assert horizon % 2 == 0, f"Horizon must be even!"
        self.horizon = horizon
        self.num_actions = num_actions

        self._state_hash = state_hash # None or int to int function
        self._action_hash = action_hash # None or int to int function

        self._basic_sample = default_sample

        self.state_space = [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])]
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
        if state == 0:
            next_state = int(2) if action == opt_action else int(1)
        elif state == 1 or state == 2:
            next_state = int(0)
        else:
            raise Exception(f"Transition: Invalid state {state}!")
        return torch.tensor([next_state])
    
    def _get_reward(self, state, action, timestep):
        '''
        Reward
        '''
        
        # Choose an optimal action for each step

        state = state.item()
        action = action.item()

        assert state == 0 or state == 1 or state == 2, f"Reward: Invalid state {state}"
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
            if action == opt_action:
                reward = 0
            elif action % 2 == 0:
                reward = 1
            else:
                reward = 2
        else: # state == 2
            if action == opt_action:
                reward = 3
            else:
                reward = 0
        
        
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