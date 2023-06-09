# Enable import of env
import sys
import os
current_path = sys.path[0] # path of generate.py
parent_search_path = os.path.join(current_path, '..') # toy directory
sys.path.append(parent_search_path)

from email import generator
import pickle
from return_transforms.algos.esper.esper import esper
# from algos.esper.esper import esper
from fire import Fire
import yaml
from pathlib import Path
import numpy as np
from env.bandit_env import BanditEnv
import torch
# import argparse


# parser = argparse.ArgumentParser()

# parser.add_argument('--data_file', type=str, default='./dataset/toy.csv')


def load_config(config_path):
    return yaml.safe_load(Path(config_path).read_text())


def normalize_obs(trajs):
    obs_list = []
    for traj in trajs:
        obs_list.extend(traj.obs)
    obs = np.array(obs_list)
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0) + 1e-8
    for traj in trajs:
        for i in range(len(traj.obs)):
            traj.obs[i] = (traj.obs[i] - obs_mean) / obs_std
    return trajs


def generate(env_name, data_file, config, ret_file, n_cpu=1):
    '''
    - env_name: str, path to env
    - data_file: str, path to data
    - config: str, path to config
    - ret_file: str, path to output estimated cluster rtg
    - n_cpu
    '''

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()

    print('Loading config...')
    config = load_config(config)

    # rep_size must be a multiple of group
    rep_size = config['cluster_model_args']['rep_size']
    groups = config['cluster_model_args']['groups']
    assert rep_size % groups == 0, f"Rep_size {rep_size} is not a multiple of groups {groups}!"

    if config['method'] == 'esper':
        print('Loading offline RL task...')
        # env, trajs = load_env(env_name)
        env = BanditEnv(env_name)
        horizon = env.get_horizon()

        # if config['normalize']:
        #     print('Normalizing observations...')
        #     trajs = normalize_obs(trajs)

        print('Creating ESPER returns...')
        rets, clusters = esper(data_file,
                     horizon,
                     env.get_action_space(),
                     config['dynamics_model_args'],
                     config['cluster_model_args'],
                     config['train_args'],
                     device,
                     n_cpu)

        # Save the returns as a pickle
        print('Saving returns...')
        Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
        with open(ret_file, 'wb') as f:
            pickle.dump([rets, clusters], f)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    Fire(generate)
