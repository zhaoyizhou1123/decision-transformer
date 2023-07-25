# Modified from Chuning's code

import os
import torch
import torch.nn as nn
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torch.utils.data.dataloader import DataLoader

from maze.algos.stitch_rcsl.models.unet import ConditionalUnet1D


class DiffusionBC:
    def __init__(self, config, dataset, logger):
        '''
        config: Contains attributes
        - act_dim
        - obs_dim
        - spectral_norm, bool. Default false
        - num_epochs
        - num_diffusion_iters
        - batch_size
        - num_workers = 1

        '''
        self.c = config
        self.dataset = dataset
        self.logger = logger
        self.device = torch.device("cuda")

        self.dataloader = DataLoader(self.dataset, shuffle=True, pin_memory=True,
                    batch_size= self.c.batch_size,
                    num_workers= self.c.num_workers)

        # Noise prediction net
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.c.act_dim,
            global_cond_dim=self.c.obs_dim,
            kernel_size=1,
            n_groups=1,
        ).to(self.device)

        if self.c.spectral_norm:
            self.noise_pred_net.apply(
                lambda x: nn.utils.spectral_norm(x)
                if isinstance(x, nn.Linear) or isinstance(x, nn.Conv1d)
                else x
            )

        self.optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6
        )

        self.lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=len(self.dataloader) * self.c.num_epochs,
        )

        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.c.num_diffusion_iters,
            # The choise of beta schedule has big impact on performance
            # We found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # Clip output to [-1,1] to improve stability
            clip_sample=True,
            # Our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )

        # Exponential moving average
        self.ema = EMAModel(model=self.noise_pred_net, power=0.75)

    def train(self):
        for epoch in range(self.c.num_epochs):
            for batch in self.dataloader:
                # (B, obs_horizon * obs_dim)
                obs = batch["obs"].to(self.device)
                # (B, pred_horizon, act_dim)
                action = batch["action"].to(self.device).unsqueeze(1)

                # Sample noise to add to actions
                noise = torch.randn(action.shape, device=self.device)

                # Sample a diffusion iteration for each data point
                batch_size = obs.shape[0]
                timesteps = torch.randint(
                    low=0,
                    high=self.noise_scheduler.config.num_train_timesteps,
                    size=(batch_size,),
                    device=self.device,
                ).long()

                # Add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps).type(torch.float32)

                # Predict the noise residual
                noise_pred = self.noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs
                )

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # Step optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Step lr scheduler every batch
                # this is different from standard pytorch behavior
                self.lr_scheduler.step()

                # Update Exponential Moving Average of the model weights
                self.ema.step(self.noise_pred_net)

                # Logging
                self.logger.record("train/loss", loss.item())

            # Dump logger
            self.logger.record("train/epoch", epoch)
            self.logger.dump()

        # Weights of the EMA model is used for inference
        self.ema_noise_pred_net = self.ema.averaged_model

    def sample_init_noise(self):
        return torch.randn((1, 1, self.c.act_dim), device=self.device)

    def select_action(self, obs, init_noise=None):
        # (B, obs_horizon * obs_dim)
        obs_cond = torch.from_numpy(obs).to(self.device, dtype=torch.float32)
        obs_cond = obs_cond.unsqueeze(0)

        # Initialize action
        if init_noise is not None:
            act_pred = init_noise.clone()
        else:
            act_pred = self.sample_init_noise()

        # Initialize scheduler
        self.noise_scheduler.set_timesteps(self.c.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.ema_noise_pred_net(
                sample=act_pred, timestep=k, global_cond=obs_cond
            )

            # Inverse diffusion step (remove noise)
            act_pred = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=act_pred
            ).prev_sample

        return act_pred.detach().cpu().numpy()[0, 0]

    def save_checkpoint(self):
        # Save checkpoint
        torch.save(
            self.ema_noise_pred_net.state_dict(),
            os.path.join(self.logger.dir, "models.pt"),
        )

    def load_checkpoint(self):
        # Load checkpoint
        ckpt_path = os.path.join(self.logger.dir, "models.pt")
        state_dict = torch.load(ckpt_path, map_location="cuda")
        self.ema_noise_pred_net = self.noise_pred_net
        self.ema_noise_pred_net.load_state_dict(state_dict)
