'''
MIT License

Copyright (c) 2022 Yi-hao Sun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

# https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/modules/dynamics_module.py

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional


class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(x)
        return x


def soft_clamp(
    x : torch.Tensor,
    _min: Optional[torch.Tensor] = None,
    _max: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # clamp tensor values while mataining the gradient
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x

class EnsembleLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_ensemble: int,
        weight_decay: float = 0.0
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble

        self.register_parameter("weight", nn.Parameter(torch.zeros(num_ensemble, input_dim, output_dim)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(num_ensemble, 1, output_dim)))

        nn.init.trunc_normal_(self.weight, std=1/(2*input_dim**0.5))

        self.register_parameter("saved_weight", nn.Parameter(self.weight.detach().clone()))
        self.register_parameter("saved_bias", nn.Parameter(self.bias.detach().clone()))

        self.weight_decay = weight_decay

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, weight)
        else:
            x = torch.einsum('bij,bjk->bik', x, weight) # batch matrix multiplication

        x = x + bias

        return x

    def load_save(self) -> None:
        self.weight.data.copy_(self.saved_weight.data)
        self.bias.data.copy_(self.saved_bias.data)

    def update_save(self, indexes: List[int]) -> None:
        self.saved_weight.data[indexes] = self.weight.data[indexes]
        self.saved_bias.data[indexes] = self.bias.data[indexes]
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = self.weight_decay * (0.5*((self.weight**2).sum()))
        return decay_loss


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]], # Union[X,Y] means either X or Y
        num_ensemble: int = 7, # Train [num_ensemble] of models simultaneously
        num_elites: int = 5, # Choose [num_elites] of models as output
        activation: nn.Module = Swish,
        weight_decays: Optional[Union[List[float], Tuple[float]]] = None,
        with_reward: bool = True,
        device: str = "cpu"
    ) -> None:
        super().__init__()

        self.num_ensemble = num_ensemble
        self.num_elites = num_elites
        self._with_reward = with_reward
        self.device = torch.device(device)

        self.activation = activation()

        assert len(weight_decays) == (len(hidden_dims) + 1)

        module_list = []
        hidden_dims = [obs_dim+action_dim] + list(hidden_dims) # Input is (obs,action)
        if weight_decays is None:
            weight_decays = [0.0] * (len(hidden_dims) + 1)
        for in_dim, out_dim, weight_decay in zip(hidden_dims[:-1], hidden_dims[1:], weight_decays[:-1]):
            module_list.append(EnsembleLinear(in_dim, out_dim, num_ensemble, weight_decay))
        self.backbones = nn.ModuleList(module_list)

        self.output_layer = EnsembleLinear(
            hidden_dims[-1],
            2 * (obs_dim + self._with_reward),
            num_ensemble,
            weight_decays[-1]
        )

        self.register_parameter(
            "max_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * 0.5, requires_grad=True)
        )
        self.register_parameter(
            "min_logvar",
            nn.Parameter(torch.ones(obs_dim + self._with_reward) * -10, requires_grad=True)
        )

        self.register_parameter(
            "elites",
            nn.Parameter(torch.tensor(list(range(0, self.num_elites))), requires_grad=False)
        )

        self.to(self.device)

    def forward(self, obs_action: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.as_tensor(obs_action, dtype=torch.float32).to(self.device)
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))
        mean, logvar = torch.chunk(self.output_layer(output), 2, dim=-1) # both (num_ensemble,batch,obs_dim+with_reward)
        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar) # clamp variance within range
        return mean, logvar

    def load_save(self) -> None:
        for layer in self.backbones:
            layer.load_save()
        self.output_layer.load_save()

    def update_save(self, indexes: List[int]) -> None:
        for layer in self.backbones:
            layer.update_save(indexes)
        self.output_layer.update_save(indexes)
    
    def get_decay_loss(self) -> torch.Tensor:
        decay_loss = 0
        for layer in self.backbones:
            decay_loss += layer.get_decay_loss()
        decay_loss += self.output_layer.get_decay_loss()
        return decay_loss

    def set_elites(self, indexes: List[int]) -> None:
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.register_parameter('elites', nn.Parameter(torch.tensor(indexes), requires_grad=False))
    
    def random_elite_idxs(self, batch_size: int) -> np.ndarray:
        idxs = np.random.choice(self.elites.data.cpu().numpy(), size=batch_size)
        return idxs