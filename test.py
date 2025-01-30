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

gym_env = gym.make('Ant-v5', render_mode="rgb_array")

agent = PPG(
    gym_env.action_space.shape[0], gym_env.observation_space.shape[0], device="cuda:1"
)

agent.load("./model.pth")

state, info = gym_env.reset(seed=seed)
episode = []
image_sequence = []

while True:
    
    image_sequence.append(gym_env.render())
    state_tensor = torch.from_numpy(state).float().to(agent.device)
    
    with torch.inference_mode():
        critic_value = agent.critic.forward(state_tensor)
        actor_dist, _ = agent.actor.forward(state_tensor)
        # sample
                        
    action_np = actor_dist.sample().cpu().view(gym_env.action_space.shape).numpy()
    next_state, reward, terminated, truncated, _ = gym_env.step(action_np)
    done = terminated | truncated
    state = next_state
    
    if done:
        gym_env.close()
        break

convert_images_to_video(image_sequence, "video.mp4", fps=30)