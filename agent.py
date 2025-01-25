import torch

from model import Actor, Critic


class Agent:

    def __init__(
        self, action_space: int, observation_space: int, normalizing_func: torch.Tensor
    ):
        self.action_space = action_space
        self.observation_space = observation_space

        self.actor = Actor(observation_space, action_space)
        self.critic = Critic(observation_space)
        self.normalizing_function = normalizing_func

    def get_action_from_policy(
        self, observation: torch.FloatTensor
    ) -> torch.distributions.Distribution:
        dist = self.actor(observation)
        return dist

    def get_log_prob(
        self, observation: torch.FloatTensor, action: torch.FloatTensor
    ) -> torch.FloatTensor:
        dist = self.actor(observation)
        return dist.log_prob(action)

    def get_value(self, observation: torch.FloatTensor) -> torch.FloatTensor:
        return self.critic(observation)

    def save(self, path: str):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path
        )

    def load(self, path: str):
        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

        self.actor.eval()
        self.critic.eval()

        return self
