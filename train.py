import gymnasium as gym
from algorithms.ppg import PPG, training_loop

# gym = gym.make("BipedalWalker-v3")

gym = gym.make('Ant-v5')

ppg_agent = PPG(gym.action_space.shape[0], gym.observation_space.shape[0], device="cuda:1")

training_loop(gym_env=gym, agent=ppg_agent, batch_size=64)
