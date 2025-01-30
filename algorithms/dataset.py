import torch
import numpy as np


class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(self, *args):
        self.data = args

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx: int):
        return tuple([d[idx] for d in self.data])
