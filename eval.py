from ppo import Actor, Critic, PPO
import torch
import numpy as np
import random

import gymnasium as gym

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

ppo = PPO(env=env)

ppo.critic.load_state_dict(torch.load("critic.pth"))
ppo.critic.eval()

ppo.actor.load_state_dict(torch.load("actor.pth"))
ppo.actor.eval()

ppo.test("imgs")
