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

        self.critic_loss = torch.nn.L1Loss()

    def reset_env(self, n_random_steps: int = 1) -> torch.FloatTensor:

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
        batch_advantages = [[] for _ in range(len(batch_rewards))]

        for episode_idx in range(len(batch_advantages)):
            rewards = batch_rewards[episode_idx]
            values = torch.cat([batch_values[episode_idx], torch.tensor([0], dtype=torch.float32)])  # Add terminal value
            advantages = torch.zeros(len(rewards), dtype=torch.float32)
            last_adv = 0

            for t in reversed(range(len(rewards))):
                delta = (rewards[t] + self.gamma * values[t + 1] - values[t]).item()
                advantages[t] = delta + self.gamma * self.gae_lambda * last_adv
                last_adv = advantages[t]

            batch_advantages[episode_idx] = advantages

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
        b_observations, b_actions, b_log_probs, b_rewards, b_values = [], [], [], [], []

        # Episodic data, keeps track of rewards per episode
        t: int = 0
        cum_rewards = []

        while t < self.timesteps_per_batch:
            current_observation = self.reset_env()
            episodic_rewards, episodic_values = [], []

            # Remaining time in the batch
            for _ in range(self.max_timesteps_per_episode):
                t += 1

                b_observations.append(current_observation)

                # Get the action and log probability
                t_obs = torch.tensor(
                    current_observation, dtype=torch.float32
                ).unsqueeze(0)

                with torch.inference_mode():

                    values = self.critic(t_obs).flatten()  # Keep as tensor
                    dist = self.actor(t_obs)

                    action = dist.rsample()
                    log_prob = dist.log_prob(action).flatten()  # Keep as tensor

                # Take a step in the environment
                action = action.flatten()
                next_observation, reward, term, trunc, _ = self.env.step(action.numpy())

                # Reward clipping
                if self.do_clip_reward:
                    reward = max(min(reward, self.clip_reward_range[1]), self.clip_reward_range[0])

                # Store the episodic rewards
                episodic_rewards.append(reward)
                episodic_values.append(values)

                b_actions.append(action)
                b_log_probs.append(log_prob)

                current_observation = next_observation

                if term or trunc or t >= self.timesteps_per_batch:
                    break

            b_rewards.append(torch.tensor(episodic_rewards, dtype=torch.float32))
            cum_rewards.append(sum(episodic_rewards))
            b_values.append(torch.tensor(episodic_values, dtype=torch.float32).flatten())

        print(f"Episode {len(b_rewards)}: {np.mean(cum_rewards)}")

        self.env.close()

        return (
            torch.tensor(b_observations, dtype=torch.float32),
            torch.stack(b_actions),
            torch.stack(b_log_probs),
            torch.cat(self.get_gae_return(b_rewards, b_values)),
            torch.cat(b_values),
        )

    def create_dataloader(
        self,
        b_observations: torch.FloatTensor,
        b_actions: torch.FloatTensor,
        b_log_probs: torch.FloatTensor,
        b_values: torch.FloatTensor,
        b_advantage: torch.FloatTensor,
        b_gt_critic: torch.FloatTensor,
    ) -> Dataset:
        data = Dataset(
            b_observations,
            b_actions,
            b_log_probs,
            b_values,
            b_advantage,
            b_gt_critic
        )
        return torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def train(self):

        for k in tqdm(range(self.n_trajectories), desc="Training"):

            if k % 100 == 0:
                self.test(f"results/{k}")

            (b_observations, b_actions, b_log_probs, b_advantages, b_values) = (
                self.rollout()
            )

            # Compute the advantage
            b_gt_critic = b_values.view(-1) + b_advantages.view(-1)

            b_normalized_advantage = (b_advantages - b_advantages.mean()) / (
                b_advantages.std() + torch.finfo(torch.float32).eps
            )

            # Losses
            loss_policy_list, loss_critic_list = [], []

            # n_epochs
            for _ in range(self.n_epochs):

                # Load dataset
                dataloader = self.create_dataloader(
                    b_observations,
                    b_actions,
                    b_log_probs,
                    b_values,
                    b_normalized_advantage,
                    b_gt_critic
                )

                for (
                    observations,
                    actions,
                    old_log_probs,
                    values,
                    n_adv,
                    gt_critic
                ) in dataloader:

                    # Compute log probabilities
                    self.actor_optimizer.zero_grad()

                    dist = self.actor(observations)
                    new_log_probs = dist.log_prob(actions)

                    n_adv = n_adv.unsqueeze(1)

                    # PPO clipped objective
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * n_adv
                    surr2 = (torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * n_adv)
                    loss_actor = (-torch.min(surr1, surr2)).mean() - 0.01 * dist.entropy().mean() # probably include a beta scheduler

                    # Update actor
                    loss_actor.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.clip_actor_grads)
                    loss_policy_list.append(loss_actor.item())
                    self.actor_optimizer.step()

                    # Critic loss
                    self.critic_optimizer.zero_grad()
                    pred_values = self.critic(observations).squeeze(-1)

                    # loss critic 
                    loss_critic = self.critic_loss(pred_values, gt_critic)

                    # Update critic
                    loss_critic.backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.clip_critic_grads)
                    loss_critic_list.append(loss_critic.item())
                    self.critic_optimizer.step()

            print("------------------------------------------------------------")
            print(f"Actor Loss: {np.mean(loss_policy_list)}\nCritic Loss: {np.mean(loss_critic_list)}")
            print("Standard deviation of rewards: ", torch.std(b_advantages).item())
            print("Actor std: ", self.actor.log_std.exp())
            print("------------------------------------------------------------")

    def test(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)

        observation = self.reset_env(0)
        frames = []

        for i in range(self.max_timesteps_per_episode):
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                dist = self.actor(observation)
                action = dist.mean
            action = action.detach().numpy()
            observation, _, term, trunc, _ = self.env.step(action.flatten())
            rgb_values = self.env.render()
            frame = Image.fromarray(rgb_values)
            frames.append(frame)

            if term | trunc:
                break

        self.env.close()

        frames[0].save(
            f"{save_path}.gif",
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
