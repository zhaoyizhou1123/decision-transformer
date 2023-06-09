import torch
import torch.nn as nn
import torch.nn.functional as F
from return_transforms.models.basic.mlp import MLP
from return_transforms.utils.utils import get_past_indices


class ClusterModel(nn.Module):
    """
    Model that takes an input trajectories in the form of obs and actions
    and returns a cluster assignment (in the form of discrete representations, of
    which there are `groups` many) for each timestep in the trajectory.

    In summary, the model performs the following steps (on trajectories that are front-padded):
    1. Concatenate the observation and action at each timestep
    2. Pass the concatenated vector through an MLP to get a representation
    3. Reverse the sequence and pass the sequence through an LSTM to get a representation
    4. Pass the LSTM hidden state through an MLP to get logits for the cluster assignment
    5. Use gumbel softmax to sample a cluster assignment

    The model also includes the return-predictor and action-predictor, which predict returns and actions
    given the cluster assignments.
    """

    def __init__(self, obs_size, action_size, rep_size, model_args, groups=4):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.rep_size = rep_size
        self.groups = groups

        self.hidden_size = model_args['obs_action_model']['hidden_size']

        self.obs_action_model = MLP(
            obs_size + action_size+1, self.hidden_size, **model_args['obs_action_model'])
        self.lstm_model = nn.LSTM(self.hidden_size,
                                  self.hidden_size,
                                  batch_first=True)
        self.logit_model = MLP(
            self.hidden_size, rep_size, **model_args['logit_model'])

        self.ret_obs_action_model = MLP(
            obs_size + action_size + 1, self.hidden_size, **model_args['ret_obs_action_model'])
        self.return_model = MLP(rep_size + self.hidden_size,
                                1, **model_args['return_model'])
        self.action_model = MLP(rep_size + obs_size,
                                action_size, **model_args['action_model'])

    def forward(self, obs, action, timesteps, seq_len, hidden=None, hard=False):
        '''
        timesteps: (batch, ctx, ). Should be horizon?
        '''
        bsz, t = obs.shape[:2] # batchsize, ctx
        obs = obs.view(bsz, t, -1)

        timesteps = timesteps.reshape(bsz,t,-1) #(batch, ctx, 1)

        # Concatenate observations and actions
        x = torch.cat([obs, action, timesteps], dim=-1)

        # Reverse the sequence in time
        x = torch.flip(x, [1]).view(bsz * t, -1) # dim -1 is (st,at,t) tuple

        # Pass through MLP to get representation
        obs_act_reps = self.obs_action_model(x).view(bsz, t, -1) # (st,at,t) representation

        # Use LSTM to get the representations for each suffix of the sequence
        if hidden is None:
            hidden = (torch.zeros(1, bsz, self.hidden_size).to(x.device),
                      torch.zeros(1, bsz, self.hidden_size).to(x.device))

        x, hidden = self.lstm_model(obs_act_reps, hidden)

        # Reverse the sequence in time again
        x = torch.flip(x, [1]).reshape(bsz * t, -1)

        # Pass through MLP to get logits for cluster assignment
        logits = self.logit_model(x) # Does each lstm(st,at,t) pair contains information from behind?

        # Some inputs are padding (0), so we mask them out
        logits[obs.view(bsz * t, -1).sum(-1) == 0] = 0

        # Sample cluster assignment
        logits = logits.view(bsz * t, self.groups, -1)
        # num_cluster = logits.shape[-1]
        clusters = F.gumbel_softmax(logits, tau=1, hard=hard)
        # clusters = F.softmax(logits, dim=-1)
        # if hard:
        #     clusters = F.one_hot(torch.argmax(clusters, dim=-1),num_cluster)
        clusters = clusters.view(bsz, t, -1)

        # ================ Compute return prediction ================
        x = torch.cat([obs, action, timesteps], dim=-1).view(bsz * t, -1)
        ret_obs_act_reps = self.ret_obs_action_model(x).view(bsz, t, -1)

        ret_input = torch.cat(
            [clusters.detach(), ret_obs_act_reps], dim=-1).view(bsz * t, -1)

        ret_pred = self.return_model(ret_input).view(bsz, t, -1)

        # ================ Compute action prediction ================

        # First, we need to get the past indices
        idxs = get_past_indices(obs_act_reps, seq_len)
        idxs = idxs.view(bsz, t, 1).expand(bsz, t, self.rep_size)

        # Get cluster representations for the past
        past_cluster = torch.gather(clusters, 1, idxs)

        obs_context = torch.cat([obs, past_cluster], dim=-1).view(bsz * t, -1)
        act_pred = self.action_model(obs_context).view(bsz, t, -1)

        return clusters, ret_pred, act_pred, hidden

    def return_preds(self, obs, action, timesteps, hard=False):
        """
        Returns the return predictions for the given trajectories.
        """
        bsz, t = obs.shape[:2]
        obs = obs.view(bsz, t, -1)
        timesteps = timesteps.reshape(bsz,t,-1) #(batch, ctx, 1)

        # Concatenate observations and actions
        x = torch.cat([obs, action, timesteps], dim=-1)

        # Reverse the sequence in time
        x = torch.flip(x, [1]).view(bsz * t, -1)

        # Pass through MLP to get representation
        obs_act_reps = self.obs_action_model(x).view(bsz, t, -1)

        # Use LSTM to get the representations for each suffix of the sequence
        hidden = (torch.zeros(1, bsz, self.hidden_size).to(x.device),
                  torch.zeros(1, bsz, self.hidden_size).to(x.device))

        x, hidden = self.lstm_model(obs_act_reps, hidden)

        # Reverse the sequence in time again
        x = torch.flip(x, [1]).reshape(bsz * t, -1)

        # Pass through MLP to get logits for cluster assignment
        logits = self.logit_model(x)

        # Some inputs are padding (0), so we mask them out
        logits[obs.view(bsz * t, -1).sum(-1) == 0] = 0

        # Sample cluster assignment
        logits = logits.view(bsz * t, self.groups, -1)
        clusters = F.gumbel_softmax(logits, tau=1, hard=hard)
        clusters = clusters.view(bsz, t, -1)

        # ================ Compute return prediction ================
        x = torch.cat([obs, action, timesteps], dim=-1).view(bsz * t, -1)
        ret_obs_act_reps = self.ret_obs_action_model(x).view(bsz, t, -1)

        ret_input = torch.cat(
            [clusters.detach(), ret_obs_act_reps], dim=-1).view(bsz * t, -1)

        ret_pred = self.return_model(ret_input).view(bsz, t, -1)

        # ================ Compute action prediction ================

        return ret_pred, clusters
