import torch
import torch.nn.functional as F

from einops import einsum

# Loss is from original PPO paper


class ClipActorLoss(torch.nn.Module):

    def __init__(self, eps: float = 0.2):
        super(ClipActorLoss, self).__init__()
        self.eps = eps

    def forward(
        self,
        old_log_prob_actions: torch.FloatTensor,
        new_log_prob_actions: torch.FloatTensor,
        advantages: torch.FloatTensor,
    ) -> torch.FloatTensor:
        ratio = (new_log_prob_actions - old_log_prob_actions).exp()

        if advantages.ndim == 1:
            advantages = advantages.unsqueeze(-1)

        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

        return -torch.min(surrogate_1, surrogate_2).mean()


# Loss is from orginal PPO paper


class EntropyActorLoss(torch.nn.Module):

    def __init__(self, weight: float = 0.05):

        super(EntropyActorLoss, self).__init__()
        self.weight = weight

    def forward(self, dist: torch.distributions) -> torch.FloatTensor:
        return -self.weight * dist.entropy().mean()


# This is somewhat weird loss, but it is from original PPO paper


class ClipCriticLoss(torch.nn.Module):

    def __init__(self, eps: float = 0.4):
        super(ClipCriticLoss, self).__init__()
        self.eps = eps
        self.loss = torch.nn.SmoothL1Loss()

    def forward(
        self,
        old_values: torch.FloatTensor,
        new_values: torch.FloatTensor,
        returns: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # Clipping the value function
        value_clipped = old_values + (new_values - old_values).clamp(
            -self.eps, self.eps
        )

        # Surrogate losses
        surrogate_1 = self.loss(new_values, returns)
        surrogate_2 = self.loss(value_clipped, returns)

        return torch.mean(torch.max(surrogate_1, surrogate_2))


# Loss is from an open-review paper


class SpectralEntropyLoss(torch.nn.Module):

    def __init__(self, weight: float = 0.02, eps: float = 1e-16, update_very: int = 4):

        super(SpectralEntropyLoss, self).__init__()

        self.eps = eps
        self.update_rate = update_very
        self.weight = weight
        self.t = 1

    def log(self, t: torch.FloatTensor) -> torch.FloatTensor:
        return t.clamp(min=self.eps).log()

    def entropy(self, prob: torch.FloatTensor) -> torch.FloatTensor:
        return (-prob * self.log(prob)).sum()

    def forward(self, model: torch.nn.Module):

        loss = torch.tensor(0.0).requires_grad_()

        for parameter in model.parameters():

            if parameter.ndim < 2:
                continue

            *_, row, col = parameter.shape
            parameter = parameter.reshape(-1, row, col)

            # Extract singular values
            singular_values = torch.linalg.svdvals(parameter)

            # Normalize singular values
            spectral_prob = singular_values.softmax(dim=-1)

            # Compute entropy
            spectral_entropy = self.entropy(spectral_prob)

            # Accumulate loss
            loss = loss + spectral_entropy

        if self.t % self.update_rate != 0:
            loss = loss * 0

        self.t += 1

        return self.weight * loss


# Loss is from PPG paper


class KLDivLoss(torch.nn.Module):

    def __init__(self, weight: float = 0.01):

        super(KLDivLoss, self).__init__()
        self.weight = weight

    def forward(self, x, y) -> torch.FloatTensor:
        return (
            self.weight * torch.nn.functional.kl_div(x, y, reduction="batchmean").mean()
        )


# Copied from Lucidrains' SimBA implementation

# def simba_orthogonal_loss(model: torch.nn.Module, simba_module: torch.nn.Module):
# loss = torch.tensor(0.0).requires_grad_()

# for module in model.modules():

#     if not isinstance(module, simba_module):
#         continue

#     weights = []

#     for layer in module.layers:
#         linear_in, linear_out = layer.branch[1], layer.branch[3]

#         weights.append(linear_in.weight.t())
#         weights.append(linear_out.weight)

#     for weight in weights:
#         norm_weight = F.normalize(weight, dim=-1)
#         cosine_dist = einsum(norm_weight, norm_weight, "i d, j d -> i j")
#         eye = torch.eye(
#             cosine_dist.shape[-1], device=cosine_dist.device, dtype=torch.bool
#         )
#         orthogonal_loss = cosine_dist[~eye].mean()
#         loss = loss + orthogonal_loss

# return loss
