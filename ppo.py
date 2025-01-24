import torch
from model import Actor, Critic

import gymnasium as gym
import random

import io

from tqdm import tqdm
from dataset import Dataset
from PIL import Image

import numpy as np
import os


class PPO:

    def __init__(
        self,
        env: gym.Env = None,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        nf_rewards: float = 1,
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
        self.timesteps_per_batch = 5000  # timesteps per batch
        self.max_timesteps_per_episode = 500  # timesteps per episode
        self.gamma = 0.99  # discount factor
        self.lam = 0.95
        self.n_epochs = 5  # number of epochs
        self.n_trajectories = 1000
        self.clip_eps = 0.2  # PPO clipping parameter
        self.nf_rewards = nf_rewards

        self.critic_loss = torch.nn.MSELoss()

    def reset_env(self, n_random_steps: int = 10) -> torch.FloatTensor:

        state, _ = self.env.reset()

        for _ in range(random.randint(0, n_random_steps)):
            random_action = self.env.action_space.sample()
            state, _, _, _, _ = self.env.step(random_action)

        return state

    def get_gae_return(
        self,
        batch_rewards: list[torch.FloatTensor],
        batch_values: list[torch.FloatTensor],
    ) -> list[torch.FloatTensor]:

        batch_advantages = [[]] * len(batch_rewards)

        for ep in range(len(batch_advantages)):

            last_adv = 0  # Advantage at terminal state is 0
            rewards = batch_rewards[ep]
            values = batch_values[ep]
            advantages = [0] * len(rewards)

            for t in reversed(range(len(rewards))):

                delta = (
                    rewards[t] + self.gamma * values[t + 1] - values[t]
                    if t < len(rewards) - 1
                    else rewards[t] - values[t]
                ).item()
                advantages[t] = delta + self.gamma * self.lam * last_adv
                last_adv = advantages[t]

            batch_advantages[ep] = torch.tensor(advantages, dtype=torch.float32)

        return batch_advantages

    def rollout(
        self,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        list[int],
    ]:
        batch_observations: list = []  # for observations
        batch_actions: list = []  # for actions
        batch_log_probs: list = []  # log probabilities
        batch_rewards: list = []  # for measuring episode returns
        batch_values: list = []  # for measuring values

        # Episodic data, keeps tracck of rewards per episode
        t: int = 0

        while t < self.timesteps_per_batch:

            observation = self.reset_env()

            episodic_rewards: list[float] = []
            episodic_values: list[float] = []

            # Remaining time in the batch
            for _ in range(self.max_timesteps_per_episode):

                t += 1

                batch_observations.append(observation)

                # Get the action and log probability
                t_obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                with torch.inference_mode():

                    values = self.critic(t_obs).flatten().tolist()
                    dist = self.actor(t_obs)

                    action = dist.rsample()
                    log_prob = dist.log_prob(action).flatten().tolist()
                    action = action.flatten().tolist()

                # Take a step in the environment
                next_obs, reward, term, trunc, _ = self.env.step(
                    action.numpy().flatten()
                )
                reward = reward

                # Store the episodic rewards
                episodic_rewards.append(reward)
                episodic_values.append(values)

                batch_actions.append(action)
                batch_log_probs.append(log_prob)

                observation = next_obs

                if term | trunc | t >= self.timesteps_per_batch:
                    break

            episodic_rewards = torch.tensor(episodic_rewards, dtype=torch.float32)
            episodic_values = torch.tensor(episodic_values, dtype=torch.float32)

            batch_rewards.append(episodic_rewards)
            batch_values.append(episodic_values)

        # Convert tensors
        batch_observations: torch.FloatTensor = torch.tensor(
            batch_observations, dtype=torch.float32
        )
        batch_actions: torch.FloatTensor = torch.tensor(
            batch_actions, dtype=torch.float32
        )
        batch_log_probs: torch.FloatTensor = torch.tensor(
            batch_log_probs, dtype=torch.float32
        )
        batch_discounted_returns: torch.FloatTensor = torch.cat(batch_values)

        batch_gae: torch.FloatTensor = torch.cat(
            self.get_gae_return(batch_rewards, batch_values)
        )
        batch_values = torch.cat(batch_values)

        self.env.close()

        return (
            batch_observations,
            batch_actions,
            batch_log_probs,
            batch_gae,
            batch_values,
        )

    def evaluate(
        self, batch_obs: torch.FloatTensor, batch_acts: torch.FloatTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        # Get the values from the critic
        V = self.critic(batch_obs).squeeze()

        # Get the distribution from the actor
        dist = self.actor(batch_obs)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs

    def train(self):

        for k in tqdm(range(self.n_trajectories), desc="Training"):

            with torch.inference_mode():
                (b_observations, b_actions, b_log_probs, b_advantages, batch_values) = (
                    self.rollout()
                )

            # Compute the advantage
            b_normalized_advantage_k = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + torch.finfo(torch.float32).eps
            )

            # Load dataset
            data = Dataset(
                b_observations,
                b_actions,
                b_log_probs,
                batch_values,
                b_normalized_advantage_k,
            )
            dataloader = torch.utils.data.DataLoader(
                data, batch_size=64, shuffle=True, drop_last=True
            )

            # Losses
            loss_policy_list, loss_critic_list = [], []

            # n_epochs
            for _ in range(self.n_epochs):

                for (
                    observations,
                    actions,
                    old_log_probs,
                    values,
                    n_advantages,
                ) in dataloader:

                    # Compute log probabilities
                    dist = self.actor(observations)
                    new_log_probs = dist.log_prob(actions)

                    # PPO clipped objective
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * n_advantages.unsqueeze(1)
                    surr2 = torch.clamp(
                        ratios, 1 - self.clip_eps, 1 + self.clip_eps
                    ) * n_advantages.unsqueeze(1)
                    loss_actor = (-torch.min(surr1, surr2)).mean()

                    # Update actor
                    self.actor_optimizer.zero_grad()
                    loss_actor.backward()
                    loss_policy_list.append(loss_actor.item())
                    self.actor_optimizer.step()

                    # Critic loss
                    pred_values = self.critic(observations).squeeze()
                    gt_critic = values + n_advantages
                    loss_critic = self.critic_loss(pred_values, gt_critic)

                    # Update critic
                    self.critic_optimizer.zero_grad()
                    loss_critic.backward()
                    loss_critic_list.append(loss_critic.item())
                    self.critic_optimizer.step()

            print(
                f"Actor Loss: {np.mean(loss_policy_list)} Critic Loss: {np.mean(loss_critic_list)}"
            )
            print("Printing the std of the log std parameter")
            print(self.actor.log_std.exp().detach().numpy())

    def test(self, save_path: str):

        obs = self.reset_env(0)

        os.makedirs(save_path, exist_ok=True)

        frames = []
        rewards = []

        for i in range(self.max_timesteps_per_episode):

            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            with torch.inference_mode():
                dist = self.actor(obs)
                action = dist.mean

            action = action.detach().numpy()

            obs, reward, term, trunc, _ = self.env.step(action.flatten())

            rgb_values = self.env.render()

            frame = Image.fromarray(rgb_values)

            rewards.append(reward)
            frames.append(frame)

            if term | trunc:
                break

        self.env.close()
        frames[0].save(
            f"{save_path}/animation.gif",
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=100,
        )


if __name__ == "__main__":

    # seed 42
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    env = gym.make("LunarLander-v3", continuous=True, render_mode="rgb_array")

    ppo = PPO(env=env)

    ppo.train()

    # save checkpoint
    torch.save(ppo.actor.state_dict(), "actor.pth")
    torch.save(ppo.critic.state_dict(), "critic.pth")
