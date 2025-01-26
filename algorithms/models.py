import torch
import torch.nn as nn
from torch.distributions import normal


class Actor(nn.Module):

    def __init__(self, n_states, n_actions):

        super(Actor, self).__init__()
        self.n_states = n_states

        self.n_actions = n_actions

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.mu = nn.Linear(in_features=64, out_features=self.n_actions)

        # Orthogonal initialization

        for layer in [self.fc1, self.fc2, self.mu]:
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0)

        self.log_std = nn.Parameter(torch.zeros(1, self.n_actions))

    def forward(self, inputs):
        x = inputs

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        mu = self.mu(x)
        dist = normal.Normal(mu, self.log_std.exp())

        return dist


class Critic(nn.Module):

    def __init__(self, n_states):

        super(Critic, self).__init__()
        self.n_states = n_states

        self.fc1 = nn.Linear(in_features=self.n_states, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.value = nn.Linear(in_features=64, out_features=1)

        # Orthogonal initialization
        for layer in [self.fc1, self.fc2, self.value]:
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, inputs):
        x = inputs
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.value(x)

        return value
