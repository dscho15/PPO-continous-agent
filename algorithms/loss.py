import torch


def clipped_value_loss(
    values: torch.FloatTensor,
    rewards: torch.FloatTensor,
    old_values: torch.FloatTensor,
    clip: float = 0.5,
) -> torch.FloatTensor:

    value_clipped = old_values + (values - old_values).clamp(-clip, clip)

    value_loss_1 = (value_clipped.flatten() - rewards) ** 2

    value_loss_2 = (values.flatten() - rewards) ** 2

    return torch.mean(torch.max(value_loss_1, value_loss_2))
