import torch


class ClipActorLoss(torch.nn.Module):

    def __init__(self, eps: float = 0.2):

        super(ClipActorLoss, self).__init__()

        self.eps = eps

    def forward(
        self,
        old_probs: torch.FloatTensor,
        new_probs: torch.FloatTensor,
        advantages: torch.FloatTensor,
    ) -> torch.FloatTensor:

        ratio = (new_probs - old_probs).exp()

        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantages

        return -torch.min(surrogate_1, surrogate_2).mean()


class EntropyActorLoss(torch.nn.Module):

    def __init__(self, beta: float = 0.01):

        super(EntropyActorLoss, self).__init__()
        self.beta = beta

    def forward(self, dist: torch.distributions) -> torch.FloatTensor:
        return -self.beta * dist.entropy().mean()


class ClipCriticLoss(torch.nn.Module):

    def __init__(self, eps: float = 0.4):

        super(ClipCriticLoss, self).__init__()
        self.eps = eps

    def forward(
        self,
        old_values: torch.FloatTensor,
        new_values: torch.FloatTensor,
        returns: torch.FloatTensor,
    ) -> torch.FloatTensor:

        value_clipped = old_values + (new_values - old_values).clamp(
            -self.eps, self.eps
        )

        value_loss_1 = (value_clipped.flatten() - returns) ** 2
        value_loss_2 = (new_values.flatten() - returns) ** 2

        return 0.5 * torch.mean(torch.max(value_loss_1, value_loss_2))


class SpectralEntropyLoss(torch.nn.Module):

    def __init__(self, beta: float = 0.02, eps: float = 1e-16, update_very: int = 4):

        super(SpectralEntropyLoss, self).__init__()

        self.eps = eps
        self.update_rate = update_very
        self.beta = beta
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

        return self.beta * loss
