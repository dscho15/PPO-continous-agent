import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import normal
from einops import reduce


class RSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):

        super(RSNorm, self).__init__()

        self.register_buffer("t", torch.tensor(1.0))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        time = self.t.item()
        mean = self.running_mean
        var = self.running_var

        normalized_x = (x - mean) / var.sqrt().clamp(min=self.eps)

        if not self.training:
            return normalized_x

        # ensure bsz is greater than 1
        if x.shape[0] > 1:

            with torch.no_grad():

                new_obs_mean = reduce(x, "... c -> c", "mean")
                delta = new_obs_mean - mean

                self.t += 1
                self.running_mean = mean + delta / time
                self.running_var = (time - 1) / time * (var + 1 / time * delta**2)

        return normalized_x


class SimBaLayer(nn.Module):

    def __init__(self, in_features: int, exp_factor: int = 1, eps: float = 1e-5):
        super(SimBaLayer, self).__init__()

        self.model = nn.Sequential(
            nn.RMSNorm(in_features, eps=eps),
            nn.Linear(in_features, in_features * exp_factor),
            nn.SiLU(),
            nn.Linear(in_features * exp_factor, in_features),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x) + x


class Actor(nn.Module):

    def __init__(
        self, n_states: int, n_actions: int, n_layers: int = 2, h_dim: int = 64
    ):

        super(Actor, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.pre_layers = nn.Sequential(
            # RSNorm(n_states),
            nn.Linear(n_states, h_dim * 2),
        )

        self.simba_layers = nn.Sequential(
            *[SimBaLayer(in_features=h_dim * 2) for _ in range(n_layers)]
        )

        self.layer_norm = nn.RMSNorm(h_dim * 2)

        self.mu = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.SiLU(), nn.Linear(h_dim, self.n_actions)
        )

        self.log_std = nn.Parameter(torch.zeros(n_actions), requires_grad=True)

        self.critic = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.SiLU(), nn.Linear(h_dim, 1)
        )

        for layer in self.pre_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x: torch.FloatTensor) -> tuple[normal.Normal, torch.FloatTensor]:

        if x.ndim != 2:
            x = x.reshape(-1, self.n_states)

        x = self.pre_layers(x)
        x = self.simba_layers(x)
        x = self.layer_norm(x)

        v = self.critic(x)

        mu = self.mu(x)
        dist = normal.Normal(mu, self.log_std.exp())

        return dist, v


class Critic(nn.Module):

    def __init__(self, n_states: int, n_layers: int = 6, h_dim: int = 64):

        super(Critic, self).__init__()

        self.n_states = n_states

        self.pre_layers = nn.Sequential(
            # RSNorm(n_states),
            nn.Linear(n_states, h_dim),
        )

        self.simba_layers = nn.Sequential(
            *[SimBaLayer(in_features=h_dim) for _ in range(n_layers)]
        )

        self.layer_norm = nn.RMSNorm(h_dim)

        self.critic = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, 1)
        )

        # init all linear layers as orthogonal
        for layer in self.pre_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        if x.ndim != 2:
            x = x.reshape(-1, self.n_states)

        x = self.pre_layers(x)
        x = self.simba_layers(x)
        x = self.layer_norm(x)

        v = self.critic(x)
        return v
