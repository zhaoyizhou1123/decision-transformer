# TODO: add timesteps dependence

from return_transforms.models.esper.cluster_model import ClusterModel
from return_transforms.models.esper.dynamics_model import DynamicsModel
from return_transforms.datasets.esper_dataset import ESPERDataset
from return_transforms.utils.utils import learned_labels
from tqdm.autonotebook import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from collections import namedtuple

import sys
sys.path.append("/gscratch/simondu/zhaoyi/decision-transformer/toy")

from env.bandit_dataset import read_data_reward

# obs, actions, ... are all list, element type depend on env
Trajectory = namedtuple(
    "Trajectory", ["obs", "actions", "rewards", "timesteps"])

def esper(data_file,
          horizon,
          action_space,
          dynamics_model_args,
          cluster_model_args,
          train_args,
          device,
          n_cpu=1,
          is_discrete = True):
    '''
    - data_file: str, path to dataset file. Different from original code
    - action_space: Tensor(num_actions, action_dim)
    '''

    # Check if discrete action space

    # if isinstance(action_space, gym.spaces.Discrete):
    #     action_size = action_space.n
    #     act_loss_fn = lambda pred, truth: F.cross_entropy(pred.view(-1, pred.shape[-1]), torch.argmax(truth, dim=-1).view(-1),
    #                                                       reduction='none')
    #     act_type = 'discrete'
    # else:
    #     action_size = action_space.shape[0]
    #     act_loss_fn = lambda pred, truth: ((pred - truth) ** 2).mean(dim=-1)
    #     act_type = 'continuous'

    if is_discrete:
        action_size = action_space.shape[0]
        # print(action_size)
        act_loss_fn = lambda pred, truth: F.cross_entropy(pred.view(-1, pred.shape[-1]), torch.argmax(truth, dim=-1).view(-1),
                                                          reduction='none')
        act_type = 'discrete'
    else:
        action_size = action_space.shape[0]
        act_loss_fn = lambda pred, truth: ((pred - truth) ** 2).mean(dim=-1)
        act_type = 'continuous'

    # Get traj from dataset. 
    states, actions, rewards, timesteps = read_data_reward(data_file, horizon)
    '''
    - states, torch.Tensor (num_trajectories, horizon+1, state_dim). Final state is fixed to 0
    - actions, (num_trajectories, horizon, action_dim). Here action_dim=1
    - rewards, (num_trajectories, horizon, 1)
    - timesteps: (num_trajectories, horizon+1). Starting timestep is adjusted to 0
    '''
    states = states[:,:-1,:] # Remove the final state
    actions = F.one_hot(actions.squeeze(dim=-1).long(), action_size)

    # timesteps = timesteps + 1 # Let timesteps start from 1, for more accurate pred
    timesteps = timesteps[:,:-1] # Remove final timestep, (num_trajectories, horizon)

    num_trajectories = states.shape[0]
    # Get trajs
    trajs = []
    for i in range(num_trajectories):
        single_traj = Trajectory(obs=states[i], 
                                 actions=actions[i],
                                 rewards=rewards[i],
                                 timesteps=timesteps[i])
        trajs.append(single_traj)


    # Get the length of the longest trajectory
    # max_len = max([len(traj.obs) for traj in trajs]) + 1

    max_len = horizon # exactly the same as horizon

    dataset = ESPERDataset(trajs, action_size, max_len,
                           gamma=train_args['gamma'], act_type=act_type)

    scale = train_args['scale']

    # Get the obs size from the first datapoint
    obs, _, _, _, _= next(iter(dataset))
    obs_shape = obs[0].shape
    obs_size = np.prod(obs_shape)

    # Set up the models
    print('Creating models...')
    dynamics_model = DynamicsModel(obs_size,
                                   action_size,
                                   cluster_model_args['rep_size'],
                                   dynamics_model_args).to(device)

    cluster_model = ClusterModel(obs_size,
                                 action_size,
                                 cluster_model_args['rep_size'],
                                 cluster_model_args,
                                 cluster_model_args['groups']).to(device)

    dynamics_optimizer = optim.AdamW(
        dynamics_model.parameters(), lr=float(train_args['dynamics_model_lr']))
    cluster_optimizer = optim.AdamW(
        cluster_model.parameters(), lr=float(train_args['cluster_model_lr']))

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_args['batch_size'],
                                             num_workers=n_cpu)

    # Calculate epoch markers
    total_epochs = train_args['cluster_epochs'] + train_args['return_epochs']
    ret_stage = train_args['cluster_epochs']

    print('Training...')

    dynamics_model.train()
    cluster_model.train()
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_act_loss = 0
        total_ret_loss = 0
        total_dyn_loss = 0
        total_baseline_dyn_loss = 0
        total_batches = 0
        for obs, acts, ret, timesteps, seq_len in pbar:
            '''
            (unclear)\n
            - obs: (batch, horizon, state_dim)
            - acts: (batch, horizon, action_dim)
            - ret: (batch, horizon)
            - timesteps: (batch, horizon)
            - seq_len: Tensor(scalar), = horizon
            '''
            total_batches += 1
            # Take an optimization step for the cluster model
            cluster_optimizer.zero_grad()
            obs = obs.to(device)
            acts = acts.to(device)
            ret = ret.to(device) / scale
            timesteps = timesteps.to(device)
            seq_len = seq_len.to(device)

            bsz, t = obs.shape[:2]

            act_mask = (acts.sum(dim=-1) == 0)
            obs_mask = (obs.view(bsz, t, -1)[:, :-1].sum(dim=-1) == 0)

            # Get the cluster predictions
            clusters, ret_pred, act_pred, _ = cluster_model(
                obs, acts, timesteps, seq_len, hard=epoch >= ret_stage)

            pred_next_obs, next_obs = dynamics_model(
                obs, acts, timesteps, clusters, seq_len)

            # Calculate the losses
            ret_loss = ((ret_pred.view(bsz, t) - ret.view(bsz, t)) ** 2).mean()
            act_loss = act_loss_fn(act_pred, acts).view(bsz, t)[
                ~act_mask].mean()
            dynamics_loss = ((pred_next_obs - next_obs) ** 2)[~obs_mask].mean()

            # Calculate the total loss
            if epoch < ret_stage:
                loss = -train_args['adv_loss_weight'] * dynamics_loss + \
                    train_args['act_loss_weight'] * act_loss
            else:
                loss = ret_loss

            loss.backward()
            cluster_optimizer.step()

            # Take an optimization step for the dynamics model
            dynamics_optimizer.zero_grad()
            pred_next_obs, next_obs = dynamics_model(
                obs, acts, timesteps, clusters.detach(), seq_len)
            baseline_pred_next_obs, _ = dynamics_model(
                obs, acts, timesteps, torch.zeros_like(clusters), seq_len)
            dynamics_loss = ((pred_next_obs - next_obs) ** 2)[~obs_mask].mean()
            baseline_dynamics_loss = (
                (baseline_pred_next_obs - next_obs) ** 2)[~obs_mask].mean()
            total_dynamics_loss = dynamics_loss + baseline_dynamics_loss
            total_dynamics_loss.backward()
            dynamics_optimizer.step()

            # Update the progress bar
            total_loss += loss.item()
            total_act_loss += act_loss.item()
            total_ret_loss += ret_loss.item()
            total_dyn_loss += dynamics_loss.item()
            total_baseline_dyn_loss += baseline_dynamics_loss.item()

            advantage = total_baseline_dyn_loss - total_dyn_loss

            pbar.set_description(
                f"Epoch {epoch} | Loss: {total_loss / total_batches:.4f} | Act: {total_act_loss / total_batches:.4f} | Ret: {total_ret_loss / total_batches:.4f} | Dyn Loss: {total_dyn_loss / total_batches:.4f} | Adv: {advantage / total_batches:.4f}")

    # Get the learned return labels
    avg_returns = []
    traj_clusters = []
    for traj in tqdm(trajs):
        labels, clusters = learned_labels(traj, cluster_model,
                                action_size, max_len, device, act_type)
        avg_returns.append(labels * scale)
        traj_clusters.append(clusters)

    return avg_returns, traj_clusters
