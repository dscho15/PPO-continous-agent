import torch

from collections import namedtuple

Memory = namedtuple(
    "Memory", ["state", "action", "action_log_prob", "reward", "done", "value"]
)

def optimize_network(
    loss: torch.FloatTensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    clip_grads: float = None,
):
    optimizer.zero_grad()
    loss.backward()
    if clip_grads is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grads)
    optimizer.step()


def to_torch_tensor(x, device, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=device)


def get_gae_advantages(
    episodes: list[Memory], gamma: float = 0.99, gae_lambda: float = 0.95
) -> list[torch.FloatTensor]:
    advantages = torch.zeros(len(episodes), dtype=torch.float32)

    episodes.append(
        Memory(
            state=None,
            action=None,
            action_log_prob=None,
            reward=None,
            done=1,
            value=0,
        )
    )
    
    states, actions, action_log_probs, rewards, dones, values = zip(*episodes)
    prev_advantage = 0
    
    for t in reversed(range(len(advantages))):
        delta_t = (rewards[t] + gamma * (~dones[t]) * values[t + 1] - values[t]).item()
        advantages[t] = delta_t + gamma * gae_lambda * prev_advantage
        prev_advantage = advantages[t]

    episodes.pop()

    return advantages


def get_cum_returns(episode: list[Memory], gamma: float = 0.95) -> torch.FloatTensor:

    cum_rewards = torch.zeros(len(episode), dtype=torch.float32)
    cum_rewards[-1] = to_torch_tensor(episode[-1].reward, cum_rewards.device)

    for t in reversed(range(len(cum_rewards) - 1)):
        reward = to_torch_tensor(episode[t].reward, cum_rewards.device)
        cum_rewards[t] = reward + gamma * cum_rewards[t + 1]

    return cum_rewards
