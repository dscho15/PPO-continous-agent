import torch
import numpy as np

from collections import namedtuple

Memory = namedtuple(
    "Memory", ["state", "action", "action_log_prob", "reward", "done", "value"]
)


def create_dataloader(
    episodes: list[list[Memory]],
    advantages: list[torch.FloatTensor],
    batch_size: int,
) -> torch.utils.data.DataLoader:

    dataset = ExperienceDataset(episodes, advantages)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        states: list[torch.FloatTensor],
        actions: list[torch.FloatTensor],
        log_probs: list[torch.FloatTensor],
        values: list[torch.FloatTensor],
        advantages: list[torch.FloatTensor],
        gt_critics: list[torch.FloatTensor],
    ):
        self.states = states
        self.actions = actions
        self.log_probs = log_probs
        self.values = values
        self.advantages = advantages
        self.gt_critics = gt_critics

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            self.states[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.values[idx],
            self.advantages[idx],
            self.gt_critics[idx],
        )


class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        episodes: list[list[Memory]],
        advantages: list[list[float]],
        epsilon: float = 1e-8,
    ):
        self.episodes = []
        for episode in episodes:
            self.episodes.extend(episode)
            
        self.advantages = []
        for advantage in advantages:
            self.advantages.extend(advantage)

        self.returns = [
            mem.value + adv for mem, adv in zip(self.episodes, self.advantages)
        ]

        self.norm_advantages = (self.advantages - np.mean(self.advantages)) / (
            np.std(self.advantages) + epsilon
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return (
            self.episodes[idx].state.flatten(),
            self.episodes[idx].action.flatten(),
            self.episodes[idx].action_log_prob.flatten(),
            self.episodes[idx].reward,
            self.episodes[idx].done,
            self.episodes[idx].value,
            self.norm_advantages[idx],
            self.returns[idx],
        )
