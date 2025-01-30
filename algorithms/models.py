import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import normal
from einops import reduce


# orthogonal initialization
def ortho_init(layer, gain=1.0):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.orthogonal_(layer.weight.data, gain=gain)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0)
        print(f"Orthogonal initialization of {layer}")
    return layer


class RSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):

        super(RSNorm, self).__init__()

        self.register_buffer("t", torch.tensor(1.0))
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        t = self.t.item()
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
                self.t.data.add_(1)
                self.running_mean.data.copy_(mean + delta * (1 / t))
                self.running_var.data.copy_(((t - 1) / t) * (var + (1 / t) * delta**2))

        return normalized_x


class SimBaLayer(nn.Module):

    def __init__(self, in_features: int, exp_factor: int = 4, eps: float = 1e-5):

        super(SimBaLayer, self).__init__()

        self.model = nn.Sequential(
            nn.RMSNorm(in_features, eps=eps),
            nn.Linear(in_features, in_features * exp_factor),
            nn.SiLU(),
            nn.Linear(in_features * exp_factor, in_features),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.model(x) + x


class SimBaEncoder(nn.Module):

    def __init__(self, n_states: int, n_layers: int = 2, h_dim: int = 64):

        super(SimBaEncoder, self).__init__()

        self.n_states = n_states

        self.pre_layers = nn.Sequential(
            # RSNorm(n_states),
            nn.Linear(n_states, h_dim),
        )

        self.simba_layers = nn.Sequential(
            *[SimBaLayer(in_features=h_dim) for _ in range(n_layers)]
        )

        self.layer_norm = nn.RMSNorm(h_dim)

        for layer in self.pre_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:

        if x.ndim != 2:
            x = x.reshape(-1, self.n_states)

        x = self.pre_layers(x)
        x = self.simba_layers(x)
        x = self.layer_norm(x)

        return x


class MlpActor(nn.Module):

    def __init__(
        self, n_states: int, n_actions: int, n_layers: int = 2, h_dim: int = 64, std_init: float = -0.5
    ):

        super(MlpActor, self).__init__()

        self.n_states = n_states
        self.n_actions = n_actions

        self.simba_encoder = SimBaEncoder(n_states, n_layers, h_dim * 2)

        self.mu = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), 
            nn.SiLU(), 
            nn.Linear(h_dim, n_actions), 
            nn.Tanh()
        )
        
        self.register_parameter("log_std", torch.nn.Parameter(torch.ones(n_actions) * std_init))
                
        self.value = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), 
            nn.SiLU(),
            nn.Linear(h_dim, 1)
        )

        for i, layer in enumerate(self.children()):
            ortho_init(layer)

    def verify_input(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim != 2:
            x = x.reshape(-1, self.n_states)
        return x

    def forward(self, x: torch.FloatTensor) -> tuple[normal.Normal, torch.FloatTensor]:
        x = self.verify_input(x)
        x = self.simba_encoder(x)
        return normal.Normal(self.mu(x), self.log_std.exp()), self.value(x)

    def distribution(self, x: torch.FloatTensor) -> normal.Normal:
        x = self.verify_input(x)
        x = self.simba_encoder(x)
        return normal.Normal(self.mu(x), self.log_std.exp())

    def sample(
        self, x: torch.FloatTensor, clamp_range: tuple[float, float] = (-1.0, 1.0)
    ) -> torch.FloatTensor:
        dist = self.distribution(x)
        return dist.rsample().clamp(*clamp_range)

    def critic(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim != 2:
            x = x.reshape(-1, self.n_states)
        x = self.simba_encoder(x)
        return self.value(x)


class MlpCritic(nn.Module):

    def __init__(self, n_states: int, n_layers: int = 6, h_dim: int = 64):

        super(MlpCritic, self).__init__()

        self.n_states = n_states

        self.simba_encoder = SimBaEncoder(n_states, n_layers, h_dim)

        self.critic = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, 1)
        )

        for i, layer in enumerate(self.children()):
            ortho_init(layer)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        if x.ndim != 2:
            x = x.reshape(-1, self.n_states)
        x = self.simba_encoder(x)
        return self.critic(x)

class CnnActor(nn.Module)