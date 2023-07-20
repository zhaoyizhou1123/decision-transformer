'''
Implement data behaviorr for point maze.
The sampling maze is similar to the true maze, but has different starting state and goals
'''

from maze.samplers.base import BaseSampler
from collections import namedtuple
from random import randint
from tqdm.autonotebook import tqdm
from copy import deepcopy
from maze.policies.maze_expert import WaypointController
from maze.utils.maze_utils import VALID_VALUE, set_map_cell
from maze.utils.trajectory import Trajectory

import gymnasium as gym
import numpy as np

'''
All elements are list type.
- 'observations': obs['observation']. 
- 'actions'
- 'rewards'
- 'returns': the return achieved, for RCSL methods
- 'timesteps': the current timestep
- 'terminated': True if achieved goal
- 'truncated': True if reached time limit
- 'infos': the info output by env
'''

class MazeSampler(BaseSampler):
    def __init__(self, horizon, maze_map, target_start, target_goal, debug = False, render = False) -> None:
        '''
        horizon: int, sampling horizon
        maze_map: list(list), map of the game, only specifies 0, 1. R and G are specified by start and goal.
        target_start, array (2,), int, the start point of the game we want to learn. Used for recording data
        target_goal, array (2,), int, the goal point of the game we want to learn
        debug, bool. If True, print debugging info
        '''
        super().__init__()
        self.horizon = horizon
        self.MAZE_MAP = deepcopy(maze_map) # The basic map, without c,r,g. Cannot be changed

        self.target_start = target_start
        self.target_goal = target_goal

        self.debug = debug
        self.render = render

        if self.debug:
            print(f"MazeSampler: Target goal {self.target_goal}")

        # target_map = set_map_cell(self.MAZE_MAP, target_goal, 'g')
        # target_map = set_map_cell(target_map, target_start, 'r')
        # self.target_env = gym.make('PointMaze_UMazeDense-v3', 
        #                            maze_map= maze_map, 
        #                            continuing_task = False,
        #                            max_episode_steps=self.horizon)

    def collect_trajectories(self, sample_args: dict):
        '''
        Sample Multiple trajectories.
        sample_args: dict, contain keys:
        - starts: list(n.array (2,))
        - goals: list(np.array (2,))

        Return: (list(Trajectory), horizon, map, target_start, target_goal). Elements 3-5 are for env
        '''
        assert 'starts' in sample_args and 'goals' in sample_args, f"sample_args is expected to have keys 'starts' and 'goals' "
        starts = sample_args['starts']
        goals = sample_args['goals']
        repeats = sample_args['repeats']
        randoms = sample_args['randoms']
        assert len(starts) == len(goals), f"collect_trajectories: starts and goals are expected to have the same length!"

        # if self.debug:
        #     print(f"Collect trajectory: starts = {starts}, goals = {goals}")

        trajs_ = []
        # print(f"Starts: {starts}")
        for idx in range(len(starts)):
            start = starts[idx]
            goal = goals[idx]
            repeat = repeats[idx]
            random_end = randoms[idx]
            trajectorys = self._collect_single_traj(start, goal, repeat, random_end)
            trajs_ += trajectorys

        return (trajs_, self.horizon, self.MAZE_MAP, self.target_start, self.target_goal)

        
    def _collect_single_traj(self, start, goal, repeat, random_end: bool):
        '''
        Collect one trajectory.
        start: np.array (2,), type=int. Initial (row,col)
        goal: np.array (2,), type=int. Goal (row,col)
        repeat: int. times to repeat
        random_end: If True, do random walk when goal reached
        Return: Trajectory
        '''

        # Configure behavior maze map, the map for controller to take action
        behavior_map = set_map_cell(self.MAZE_MAP, start, 'r')
        behavior_map = set_map_cell(behavior_map, goal, 'g')

        # Set up behavior environment
        if self.render:
            render_mode = 'human'
        else:
            render_mode = None
        # render_mode = None

        # print(f"Behavior_env render mode {render_mode}")
        behavior_env = gym.make('PointMaze_UMazeDense-v3', 
                              maze_map= behavior_map, 
                              continuing_task = False,
                              max_episode_steps=self.horizon,
                              render_mode = render_mode)

        # Set up controller
        controller = WaypointController(maze = deepcopy(behavior_env.maze))

        # Configure data maze and env, the env to record data
        data_map = set_map_cell(self.MAZE_MAP, start, 'r')
        data_map = set_map_cell(data_map, self.target_goal, 'g')

        data_env = gym.make('PointMaze_UMazeDense-v3', 
                              maze_map= data_map, 
                              continuing_task = False,
                              max_episode_steps=self.horizon)
        
        # if self.debug:
        #     print(f"behavior_env==data_env: {behavior_env==data_env}")
        
        # Initialize data to record
        trajs = []
        for _ in range(repeat):
            observations_ = []
            actions_ = []
            rewards_ = []
            achieved_rets_ = [] # The total reward that has achieved, used to compute rtg
            timesteps_ = []
            terminateds_ = []
            truncateds_ = []
            infos_ = []

            # reset, data_env, behavior_env only differ in reward
            seed = np.random.randint(0, 1000)
            behavior_obs, _ = behavior_env.reset(seed=seed)
            obs, _ = data_env.reset(seed=seed)
            if self.debug:
                print(f"True goal: {obs['desired_goal']}, Sample goal: {behavior_obs['desired_goal']}")

            # Initialize return accumulator, terminated, truncated, info
            achieved_ret = 0
            data_terminated = False
            behavior_terminated = False
            truncated = False
            info = None

            for n_step in range(self.horizon):
                observations_.append(deepcopy(obs['observation']))
                achieved_rets_.append(deepcopy(achieved_ret))
                timesteps_.append(deepcopy(n_step))
                terminateds_.append(data_terminated) # We assume starting point is unfinished
                truncateds_.append(truncated)
                infos_.append(info)

                # if data_terminated: # Reached true goal, don't move, dummy action, change reward to 1
                #     # action = np.zeros(2,)
                #     # reward = 1

                #     # Continue control
                #     pass
                    
                # else: 
                    # controller uses the 'desired_goal' key of obs to know the goal, not the goal mark on the map
                if behavior_terminated and random_end: # sample goal reached, take no action
                    action = behavior_env.action_space.sample()
                else: # still head toward the sample goal
                    action = controller.compute_action(behavior_obs)
                # action = controller.compute_action(behavior_obs)

                behavior_obs, _, behavior_terminated, _, _ = behavior_env.step(action)
                if self.debug:
                    print(f"Step {n_step}, behavior maze, current pos {behavior_obs['achieved_goal']}, terminated {behavior_terminated}")

                obs, reward, data_terminated, truncated, info = data_env.step(action)
                if self.debug:
                    print(f"Step {n_step}, data maze, current pos {obs['achieved_goal']}, terminated {data_terminated}, reward {reward}")

                actions_.append(deepcopy(action))
                rewards_.append(deepcopy(reward))

                # Update return
                achieved_ret += reward

            # Compute returns. Note that achieved_ret is now total return
            total_ret = achieved_ret
            returns_ = [total_ret - achieved for achieved in achieved_rets_]
            trajs.append(Trajectory(observations = observations_, 
                          actions = actions_, 
                          rewards = rewards_, 
                          returns = returns_, 
                          timesteps = timesteps_, 
                          terminated = terminateds_, 
                          truncated = truncateds_, 
                          infos = infos_))
        behavior_env.close()
        data_env.close()

            # if data_terminated:
            #     print(f"Warning: data_env already reached goal, quit sampling immediately")
            #     break
            # if behavior_terminated:
            #     print(f"Behavior env finished")
            #     break



        return trajs
    
    def get_expert_return(self, repeat=10):
        '''
        Collect one trajectory.
        start: np.array (2,), type=int. Initial (row,col)
        goal: np.array (2,), type=int. Goal (row,col)
        repeat: int. times to repeat
        random_end: If True, do random walk when goal reached
        Return: Trajectory
        '''

        # Configure behavior maze map, the map for controller to take action
        behavior_map = set_map_cell(self.MAZE_MAP, self.target_start, 'r')
        behavior_map = set_map_cell(behavior_map, self.target_goal, 'g')

        # Set up behavior environment
        if self.render:
            render_mode = 'human'
        else:
            render_mode = None
        # render_mode = None

        # print(f"Behavior_env render mode {render_mode}")
        behavior_env = gym.make('PointMaze_UMazeDense-v3', 
                              maze_map= behavior_map, 
                              continuing_task = False,
                              max_episode_steps=self.horizon,
                              render_mode = render_mode)

        # Set up controller
        controller = WaypointController(maze = deepcopy(behavior_env.maze))
        
        # if self.debug:
        #     print(f"behavior_env==data_env: {behavior_env==data_env}")
        
        # Initialize data to record
        rets = []
        for epoch in range(repeat):

            # reset, data_env, behavior_env only differ in reward
            seed = np.random.randint(0, 1000)
            behavior_obs, _ = behavior_env.reset(seed=seed)
            if self.debug:
                print(f"Goal: {behavior_obs['desired_goal']}")

            # Initialize return accumulator, terminated, truncated, info
            achieved_ret = 0

            for n_step in range(self.horizon):

                # if data_terminated: # Reached true goal, don't move, dummy action, change reward to 1
                #     # action = np.zeros(2,)
                #     # reward = 1

                #     # Continue control
                #     pass
                    
                # else: 
                    # controller uses the 'desired_goal' key of obs to know the goal, not the goal mark on the map
                action = controller.compute_action(behavior_obs)
                # action = controller.compute_action(behavior_obs)

                behavior_obs, reward, behavior_terminated, _, _ = behavior_env.step(action)
                if self.debug:
                    print(f"Step {n_step}, behavior maze, current pos {behavior_obs['achieved_goal']}, terminated {behavior_terminated}")

                # Update return
                achieved_ret += reward

            # Compute returns. Note that achieved_ret is now total return
            print(f"Epoch {epoch}, total return {achieved_ret}")
            rets.append(achieved_ret)

        behavior_env.close()

            # if data_terminated:
            #     print(f"Warning: data_env already reached goal, quit sampling immediately")
            #     break
            # if behavior_terminated:
            #     print(f"Behavior env finished")
            #     break
        return sum(rets) / len(rets)




