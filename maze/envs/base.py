'''
Modified from https://github.dev/keirp/stochastic_offline_envs
'''

from maze.samplers.trajectory_sampler import TrajectorySampler
from os import path
import pickle


class BaseOfflineEnv:
    '''
    Encapsulates env and dataset
    '''
    def __init__(self, data_path, env_cls, data_policy_fn, horizon, n_interactions):
        '''
        data_path: str, path to dataset file
        env_cls: function, return the environment
        data_policy_fn: A function that returns BasePolicy class, can reset/sample/update, as data sampling policy. 
        horizon: int, horizon of each episode
        n_interactions: int, max number of interactions with env. episode num = n_interactions / horizon
        n_interactions
        '''
        self.env_cls = env_cls
        self.data_policy_fn = data_policy_fn
        self.horizon = horizon
        self.n_interactions = n_interactions
        self.data_path = data_path
        if self.data_path is not None and path.exists(self.data_path):
            print('Dataset file found. Loading existing trajectories.')
            with open(self.data_path, 'rb') as file:
                self.trajs = pickle.load(file)
        else:
            print('Dataset file not found. Generating trajectories.')
            self.generate_and_save()

    def generate_and_save(self):
        self.trajs = self.collect_trajectories()

        if self.data_path is not None:
            with open(self.data_path, 'wb') as file:
                pickle.dump(self.trajs, file)
                print('Saved trajectories to dataset file.')

    def collect_trajectories(self):
        data_policy = self.data_policy_fn()
        sampler = TrajectorySampler(env_cls=self.env_cls,
                                    policy=data_policy,
                                    horizon=self.horizon)
        trajs = sampler.collect_trajectories(self.n_interactions)
        return trajs


def default_path(name):
    # Get the path of the current file
    file_path = path.dirname(path.realpath(__file__))
    # Go up 3 directories
    root_path = path.abspath(path.join(file_path, '..', '..', '..'))
    # Go to offline data directory
    offline_data_path = path.join(root_path, 'offline_data')
    # append the name of the dataset
    return path.join(offline_data_path, name)
