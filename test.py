from algorithms.ppg import PPG
from tqdm import tqdm
from algorithms.utils import convert_images_to_video

import torch
import numpy as np
import random

import gymnasium as gym


seed = 42
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

gym_env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

agent = PPG(
    gym_env.action_space.shape[0], gym_env.observation_space.shape[0]
)

agent.load("./model.pth")

state, info = gym_env.reset(seed=seed)
episode = []
image_sequence = []

while True:
    
    image_sequence.append(gym_env.render())

    state_tensor = torch.from_numpy(state).float().to(agent.device)
    
    with torch.inference_mode():
    
        critic_value = agent.ema_critic.forward_eval(state_tensor)
        actor_dist, _ = agent.ema_actor.forward_eval(state_tensor)
        action = actor_dist.mean
        log_prob_action = actor_dist.log_prob(action)

    action_np = action.cpu().view(gym_env.action_space.shape).numpy()
    next_state, reward, terminated, truncated, _ = gym_env.step(action_np)

    done = terminated | truncated

    state = next_state
    
    if done:
        gym_env.close()
        break

convert_images_to_video(image_sequence, "video.mp4", fps=30)