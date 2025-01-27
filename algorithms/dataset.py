import torch
import numpy as np
from algorithms.utils import get_gae_advantages

from collections import namedtuple

Memory = namedtuple(
    "Memory", ["state", "action", "action_log_prob", "reward", "done", "value"]
)

MemoryAux = namedtuple(
    "MemoryAux", ["state", "returns", "action_log_prob"]
)

class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[list[Memory]]
    ):
        self.advantages = []
        self.episodes = []
        for e in episodes:
            self.advantages.extend(get_gae_advantages(e))
            self.episodes.extend(e)
            
        (self.states, self.actions, self.actions_log_prob, self.rewards, self.done, self.values) = zip(*self.episodes)
        
        self.returns = [value + adv for value, adv in zip(self.values, self.advantages)]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx].flatten(),
            self.actions[idx].flatten(),
            self.actions_log_prob[idx].flatten(),
            self.rewards[idx],
            self.done[idx],
            self.values[idx],
            self.advantages[idx],
            self.returns[idx],
        )
        
class ExperienceAuxDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes_aux: list[MemoryAux]
    ):
        self.episodes_aux = episodes_aux
        
        (self.states, self.returns, self.actions_log_prob) = zip(*self.episodes_aux)
        
        

    def __len__(self):
        return len(self.episodes_aux)

    def __getitem__(self, idx):
        return (
            self.episodes_aux[idx].state.flatten(),
            self.episodes_aux[idx].returns,
            self.episodes_aux[idx].old_value
        )
