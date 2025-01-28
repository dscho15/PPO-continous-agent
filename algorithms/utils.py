import torch
from collections import namedtuple
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

Memory = namedtuple(
    "Memory", ["state", "action", "action_log_prob", "reward", "done", "value"]
)


def update_network(
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

    _, _, _, rewards, dones, values = zip(*episodes)
    prev_advantage = 0

    for t in reversed(range(len(advantages))):
        delta_t = (rewards[t] + gamma * (~dones[t]) * values[t + 1] - values[t]).item()
        advantages[t] = delta_t + gamma * gae_lambda * prev_advantage
        prev_advantage = advantages[t]

    episodes.pop()

    return advantages


def get_cum_returns_exact(
    episode: list[Memory], gamma: float = 0.95
) -> torch.FloatTensor:

    returns = torch.zeros(len(episode), dtype=torch.float32)
    returns[-1] = to_torch_tensor(episode[-1].reward, returns.device)

    for t in reversed(range(len(returns) - 1)):
        reward = to_torch_tensor(episode[t].reward, returns.device)
        returns[t] = reward + gamma * returns[t + 1]

    return returns


def get_returns(values: list, advantages: list):
    returns = [value + adv for value, adv in zip(values, advantages)]
    return to_torch_tensor(returns, "cpu").view(-1, 1)


def convert_images_to_video(image_list, output_file, fps=30):
    if not image_list:
        raise ValueError("The image list is empty.")

    # Create a video clip from the image list
    clip = ImageSequenceClip(image_list, fps=fps)

    # Write the video to the specified output file
    clip.write_videofile(output_file, codec="libx264", audio=False)
