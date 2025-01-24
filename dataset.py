import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        observations: list[torch.FloatTensor],
        actions: list[torch.FloatTensor],
        log_probs: list[torch.FloatTensor],
        values: list[torch.FloatTensor],
        advantages: list[torch.FloatTensor],
    ):
        self.observations = observations
        self.actions = actions
        self.log_probs = log_probs
        self.values = values
        self.advantages = advantages

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.log_probs[idx],
            self.values[idx],
            self.advantages[idx],
        )
