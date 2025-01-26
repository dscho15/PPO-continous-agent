import torch
import torch.nn as nn
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

        with torch.no_grad():

            new_obs_mean = reduce(x, "... c -> c", "mean")
            delta = new_obs_mean - mean

            self.t += 1
            self.running_mean = mean + delta / time
            self.running_var = (time - 1) / time * (var + 1 / time * delta**2)

        return normalized_x


class ReluSquared(torch.nn.Module):

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x.sign() * torch.nn.functional.relu(x) ** 2


class SimBaLayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        exp_factor: int = 2,
        eps: float = 1e-5,
        dropout: float = 0.1,
    ):
        super(SimBaLayer, self).__init__()

        self.model = nn.Sequential(
            nn.LayerNorm(in_features, eps=eps),
            nn.Linear(in_features, in_features * exp_factor),
            ReluSquared(),
            nn.Linear(in_features * exp_factor, in_features),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x) + x


class Actor(nn.Module):

    def __init__(
        self, n_states: int, n_actions: int, n_layers: int = 6, h_dim: int = 64
    ):

        super(Actor, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.pre_layers = nn.Sequential(
            RSNorm(n_states),
            nn.Linear(n_states, h_dim * 2),
        )

        self.simba_layers = nn.Sequential(
            *[SimBaLayer(in_features=h_dim * 2) for _ in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(h_dim * 2)

        self.mu = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), ReluSquared(), nn.Linear(h_dim, self.n_actions)
        )
        self.log_std = nn.Parameter(torch.zeros(n_actions))

        self.critic = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), ReluSquared(), nn.Linear(h_dim, 1)
        )

    def forward(self, x: torch.FloatTensor) -> tuple[normal.Normal, torch.FloatTensor]:

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
            RSNorm(n_states),
            nn.Linear(n_states, h_dim),
        )

        self.simba_layers = nn.Sequential(
            *[SimBaLayer(in_features=h_dim) for _ in range(n_layers)]
        )

        self.layer_norm = nn.LayerNorm(h_dim)

        self.critic = nn.Sequential(
            nn.Linear(h_dim, h_dim), ReluSquared(), nn.Linear(h_dim, 1)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        x = self.pre_layers(x)
        x = self.simba_layers(x)
        x = self.layer_norm(x)

        v = self.critic(x)

        return v


# # test
actor = Actor(4, 2)
critic = Critic(4)

observations = torch.randn(32, 4)

dist, values = actor(observations)
dist
values.shape
values = critic(observations)
values.shape
