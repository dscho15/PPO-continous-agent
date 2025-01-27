import gymnasium as gym
from algorithms.ppg import PPG, training_loop

gym = gym.make("BipedalWalker-v3")

ppg_agent = PPG(gym.observation_space.shape[0], gym.action_space.shape[0])

training_loop(gym=gym,
              agent=ppg_agent,
              batch_size=64)
