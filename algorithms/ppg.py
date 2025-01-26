import torch
from algorithms.models import Actor, Critic

import gymnasium as gym
import random

import io

from tqdm import tqdm
from dataset import Dataset
from PIL import Image

import numpy as np
import os

from collections import namedtuple

Memory = namedtuple(
    "Memory", ["observation", "action", "action_log_prob", "reward", "done", "value"]
)


class ExperienceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        advantages: list[list[float]],
        episodes: list[list[Memory]],
        epsilon: float = 1e-8,
    ):
        self.episodes = []
        for episode in episodes:
            self.episodes += episode

        self.advantages = []
        for adv in advantages:
            self.advantages += adv

        self.gt_critic_values_ = []
        for mem, adv in zip(self.episodes, self.advantages):
            self.gt_critic_values_.append(mem.value + adv)

        self.normalized_advantages = (self.advantages - np.mean(self.advantages)) / (
            np.std(self.advantages) + epsilon
        )

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return (
            self.episodes[idx].observation.flatten(),
            self.episodes[idx].action.flatten(),
            self.episodes[idx].action_log_prob.flatten(),
            self.episodes[idx].reward,
            self.episodes[idx].done,
            self.episodes[idx].value,
            self.normalized_advantages[idx],
            self.gt_critic_values_[idx],
        )


class PPG(object):

    def __init__(
        self,
        env: gym.Env = None,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        nf_rewards: float = 1,
        clip_critic_grads: float = 0.5,
        clip_actor_grads: float = 0.5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        n_epochs: int = 10,
        n_trajectories: int = 1000,
        timesteps_per_batch: int = 2500,
        max_timesteps_per_episode: int = 1000,
        clip_eps: float = 0.2,
        batch_size: int = 64,
        do_clip_reward: bool = False,
        clip_reward_range: tuple[float, float] = (-10, 10),
    ):
        self.env = env
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]

        # Actor and Critic networks
        self.actor = Actor(self.observation_space, self.action_space)
        self.critic = Critic(self.observation_space)

        # Initialize optimizer
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=lr_critic
        )

        # Default values for hyperparameters, will need to change later.
        self.batch_size = batch_size
        self.clip_actor_grads = clip_actor_grads
        self.clip_critic_grads = clip_critic_grads
        self.clip_eps = clip_eps

        self.clip_reward_range = clip_reward_range
        self.do_clip_reward = do_clip_reward

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.n_epochs = n_epochs
        self.n_trajectories = n_trajectories
        self.nf_rewards = nf_rewards
        self.timesteps_per_batch = timesteps_per_batch
        self.beta = 0.01

        self.critic_loss = torch.nn.L1Loss()

    def reset_env(self, n_random_steps: int = 1) -> torch.FloatTensor:

        state, _ = self.env.reset()

        for _ in range(random.randint(0, n_random_steps)):

            random_action = self.env.action_space.sample()

            state, _, _, _, _ = self.env.step(random_action)

        return state

    def get_advantages(
        self,
        episode: list[Memory],
    ) -> list[torch.FloatTensor]:

        prev_adv = 0
        adv = torch.zeros(len(episode), dtype=torch.float32)
        episode.append(
            Memory(
                observation=None,
                action=None,
                action_log_prob=None,
                reward=None,
                done=1,
                value=0,
            )
        )

        for t in reversed(range(len(adv))):
            delta_t = (
                episode[t].reward
                + self.gamma * (~episode[t].done) * episode[t + 1].value
                - episode[t].value
            ).item()

            adv[t] = delta_t + self.gamma * self.gae_lambda * prev_adv

            prev_adv = adv[t]

        episode.pop()
        return adv

    def get_cumulative_rewards(self, episode: list[Memory]) -> list[torch.FloatTensor]:

        cumulative_rewards = torch.zeros(len(episode), dtype=torch.float32)
        cumulative_rewards[-1] = episode[-1].reward

        for t in reversed(range(len(cumulative_rewards) - 1)):
            cumulative_rewards[t] = (
                episode[t].reward + self.gamma * cumulative_rewards[t + 1]
            )

        return cumulative_rewards

    def rollout(
        self,
    ) -> list[list[Memory]]:

        t: int = 0
        episodes: list[list[Memory]] = []

        while t < self.timesteps_per_batch:

            s = self.reset_env()
            episode: list[Memory] = []

            for _ in range(self.max_timesteps_per_episode):

                with torch.inference_mode():

                    s = torch.tensor(s, dtype=torch.float32)

                    dist = self.actor(s)
                    a = dist.sample()
                    log_a = dist.log_prob(a)

                    v = self.critic(s)

                s_next, reward, terminated, truncated, info = self.env.step(
                    a.numpy().flatten()
                )

                mem = Memory(
                    observation=s,
                    action=a,
                    action_log_prob=log_a,
                    reward=reward,
                    done=terminated or truncated,
                    value=v,
                )
                episode.append(mem)

                s = s_next
                t += 1

                if t >= self.timesteps_per_batch or mem.done:
                    break

            episodes.append(episode)

        return episodes

    def create_dataloader(
        self, episodes: list[list[Memory]], advantages: list[torch.FloatTensor]
    ) -> Dataset:

        dataset = ExperienceDataset(advantages, episodes)

        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def train(self):

        for traj in tqdm(range(self.n_trajectories), desc="Trajectories"):

            episodes = self.rollout()

            advantages = []
            for episode in episodes:
                advantages.append(self.get_advantages(episode))

            disc_cum_rewards = []
            for episode in episodes:
                disc_cum_rewards.append(self.get_cumulative_rewards(episode))

            dataloader = self.create_dataloader(episodes, advantages)

            for epoch in range(self.n_epochs):

                for batch in dataloader:

                    (
                        observations,
                        actions,
                        action_log_probs,
                        rewards,
                        dones,
                        values,
                        advantages,
                        gt_critic_values,
                    ) = batch

                    if advantages.shape != (self.batch_size, 1):
                        advantages = advantages.unsqueeze(1)

                    dist = self.actor(observations)
                    log_probs = dist.log_prob(actions)

                    ratio = (log_probs - action_log_probs).exp()

                    self.actor_optimizer.zero_grad()

                    surrogate_1 = ratio * advantages
                    surrogate_2 = (
                        torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                        * advantages
                    )

                    actor_clip = -torch.min(surrogate_1, surrogate_2).mean()
                    actor_entropy = dist.entropy().mean()

                    actor_loss = actor_clip - self.beta * actor_entropy

                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), self.clip_actor_grads
                    )
                    actor_loss.backward()


if __name__ == "__main__":

    ppg_agent = PPG(gym.make("LunarLanderContinuous-v3"))

    ppg_agent.train()
