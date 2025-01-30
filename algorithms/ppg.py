from tqdm import tqdm
from algorithms.models import MlpActor, MlpCritic
from algorithms.dataset import ExperienceDataset
from functools import partial

from algorithms.utils import (
    update_network,
    to_torch_tensor,
    get_gae_advantages,
    get_returns,
    get_cum_returns_exact,
    normalize,
    Memory,
    MemoryAux,
)

from algorithms.losses import (
    ClipActorLoss,
    EntropyActorLoss,
    ClipCriticLoss,
    SpectralEntropyLoss,
    KLDivLoss,
)

from algorithms.ema import EMA
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from collections import deque
from torch.utils.data import Dataset

import gymnasium as gym
import numpy as np
import torch

from typing import Callable


def create_shuffled_dataloader(
    dataset: ExperienceDataset = None, batch_size: int = 64, collate_fn: Callable = None
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def batch_on_device(batch: list, device: torch.device):
    batch = [torch.stack(samples).to(device) for samples in zip(*batch)]
    return batch


def divisible_by(x: int, y: int) -> bool:
    return x % y == 0


class PPG(object):

    def __init__(
        self,
        dim_action_space: int,
        dim_obs_space: int,
        actor_kl_beta: float = 0.01,
        clip_actor_eps: float = 0.2,
        clip_actor_grads: float = 1.0,
        clip_critic_eps: float = 0.4,
        clip_critic_grads: float = 1.0,
        device: str = "cuda:0",
        ema_decay: float = 0.90,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        n_epochs: int = 2,
        std_init: float = -0.5,
        n_trajectories: int = 1000,
        save_path: str = "./model.pth",
        batch_size: int = 64,
        h_dim: int = 64,
        actor_n_layers: int = 2,
        critic_n_layers: int = 6,
    ):
        self.observation_space = dim_obs_space
        self.action_space = dim_action_space
        self.save_path = save_path
        self.batch_size = batch_size

        self.actor = MlpActor(self.observation_space, self.action_space, n_layers=actor_n_layers, h_dim=h_dim, std_init=std_init).to(device)
        self.critic = MlpCritic(self.observation_space, n_layers=critic_n_layers).to(device)

        self.opt_actor = torch.optim.AdamW(lr=lr_actor, params=self.actor.parameters())
        self.opt_critic = torch.optim.AdamW(lr=lr_critic, params=self.critic.parameters())

        self.clip_actor_loss = ClipActorLoss(clip_actor_eps)
        self.entropy_actor_loss = EntropyActorLoss(actor_kl_beta)
        self.clip_critic_loss = ClipCriticLoss(clip_critic_eps)

        self.spec_entropy_actor_loss = SpectralEntropyLoss(0.1)
        self.spec_entropy_critic_loss = SpectralEntropyLoss(0.1)
        self.kl_div_loss = KLDivLoss(1.0)

        self.clip_actor_grads = clip_actor_grads
        self.clip_critic_grads = clip_critic_grads
        self.clip_eps = clip_actor_eps
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.n_trajectories = n_trajectories

        self.actor.to(self.device)
        self.critic.to(self.device)

    def save(self):
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            str(self.save_path),
        )

    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])

    def mode(self, mode: str):
        modes = {
            "train": lambda: (self.actor.train(), self.critic.train()),
            "eval": lambda: (self.actor.eval(), self.critic.eval()),
        }

        if mode in modes:
            modes[mode]()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'eval'.")

    def learn(
        self, episodes: deque[list[Memory]], episodes_aux: deque[MemoryAux]
    ) -> tuple[list[float]]:
        _episodes = []
        advantages = []
        normalized_episode_advantages = []

        for e in episodes:
            _episodes.extend(e)
            episode_advantage = get_gae_advantages(e, self.gamma, self.gae_lambda).view(-1)
            advantages.extend(episode_advantage)
            normalized_episode_advantages.extend(episode_advantage)
            
        states, actions, action_log_probs, _, _, values = zip(*_episodes)
        returns = [adv + value for adv, value in zip(advantages, values)]

        size = len(states)
        states = torch.stack(states).view(size, -1)
        actions = torch.stack(actions).view(size, -1)
        action_log_probs = torch.stack(action_log_probs).view(size, -1)
        values = torch.stack(values).view(size, -1)
        normalized_episode_advantages = normalize(torch.stack(normalized_episode_advantages)).view(size, -1)
        returns = torch.stack(returns).view(size, -1)

        dataset = ExperienceDataset(
            states,
            actions,
            action_log_probs,
            values,
            normalized_episode_advantages,
            returns,
        )

        dataloader = create_shuffled_dataloader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=partial(batch_on_device, device=self.device),
        )

        epochs_actor_losses, epochs_critic_losses = [], []

        for i in range(self.n_epochs):
            
            last_epoch = True if i == self.n_epochs - 1 else False
            actor_losses, critic_losses = [], []

            for batch in dataloader:

                (
                    states,
                    actions,
                    old_actions_log_probs,
                    old_values,
                    normalized_episode_advantages,
                    returns,
                ) = batch

                actor_dist = self.actor.distribution(states)
                new_actions_log_probs = actor_dist.log_prob(actions)

                actor_loss = (
                    self.clip_actor_loss(
                        old_actions_log_probs,
                        new_actions_log_probs,
                        normalized_episode_advantages,
                    )
                    + self.spec_entropy_actor_loss(self.actor)
                    + self.entropy_actor_loss(actor_dist)
                )

                update_network(
                    actor_loss, self.actor, self.opt_actor, self.clip_actor_grads
                )

                actor_losses.append(actor_loss.item())

                new_values = self.critic(states)

                critic_loss = self.clip_critic_loss(old_values, new_values, returns).mean() + self.spec_entropy_critic_loss(self.critic)

                update_network(
                    critic_loss, self.critic, self.opt_critic, self.clip_critic_grads
                )

                critic_losses.append(critic_loss.item())

                if last_epoch:
                    episodes_aux.append(MemoryAux(states, actions, old_values, returns))

            epochs_actor_losses.append(np.mean(actor_losses))
            epochs_critic_losses.append(np.mean(critic_losses))

        return epochs_actor_losses, epochs_critic_losses

    def learn_aux(self, aux_episodes: deque[MemoryAux]):

        dataset = ExperienceDataset(aux_episodes)

        states, actions, old_values, returns = zip(*aux_episodes)

        states = torch.stack(states).flatten(0, 1).to(self.device)
        actions = torch.stack(actions).flatten(0, 1).to(self.device)
        old_values = torch.stack(old_values).flatten(0, 1).to(self.device)
        returns = torch.stack(returns).flatten(0, 1).to(self.device)

        with torch.inference_mode():
            actor_dist = self.actor.distribution(states)
            old_actor_log_probs = actor_dist.log_prob(actions)

        dataset = ExperienceDataset(
            states, actions, returns, old_values, old_actor_log_probs
        )

        dataloader = create_shuffled_dataloader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=partial(batch_on_device, device=self.device)
        )

        for n in range(self.n_epochs * 4):

            for batch in dataloader:

                (
                    states, 
                    actions, 
                    returns, 
                    old_values, 
                    old_log_prob_action
                ) = batch

                actor_dist_new, policy_values = self.actor(states)
                new_log_prob_action = actor_dist_new.log_prob(actions)

                loss = self.clip_critic_loss(
                    old_values, policy_values, returns
                ) + self.kl_div_loss(
                    new_log_prob_action, 
                    old_log_prob_action
                )

                update_network(loss, self.actor, self.opt_actor, self.clip_actor_grads)

                values = self.critic(states)

                loss = self.clip_critic_loss(old_values, values, returns)

                update_network(loss, self.critic, self.opt_critic, self.clip_critic_grads)


def evaluate_policy(gym_env, agent: PPG, episodes=3, render=False, seed=42):
    total_rewards = []

    for _ in range(episodes):
        state, _ = gym_env.reset(seed=seed)
        done = False
        episode_reward = 0

        while not done:
            
            if render:
                gym_env.render()
            
            with torch.inference_mode():

                state_tensor = torch.from_numpy(state).float().to(agent.device)
                agent.actor(state_tensor)
                action_dist = agent.actor.distribution()
                
            action_np = action_dist.mean.cpu().view(gym_env.action_space.shape).numpy()
            obs, reward, done, _ = gym_env.step(action_np)

            episode_reward += reward

        total_rewards.append(episode_reward)

    mean_reward = np.mean(total_rewards)
    print(f"Mean Reward over {episodes} episodes: {mean_reward:.2f}")
    return mean_reward


def training_loop(
    gym_env: gym.Env,
    agent: PPG,
    n_training_loops: int = 10000,
    max_steps: int = 3000,
    steps_before_update: int = 5000,
    steps_before_eval: int = 25000,
    num_policy_iterations_before_aux: int = 4,
    seed: int = 42,
    batch_size: int = 64,
    reward_scaling: int = 0.10,
    mode: str = "ppo" # didnt get ppg to work yet
):
    episodes = deque()
    aux_episodes = deque()
    step_count = 1
    num_policy_updates = 0
    best_reward = -np.inf
    accum_episode_rewards = []
    agent.batch_size = batch_size

    for i in tqdm(range(n_training_loops), desc="Training Loop", position=0):

        state, _ = gym_env.reset(seed=seed)

        episode = []
        accum_episode_reward = 0
        n_steps = 0

        while True:
            
            n_steps += 1
            state_tensor = torch.from_numpy(state).float().to(agent.device)

            with torch.inference_mode():
                
                critic_value = agent.critic(state_tensor)
                actor_dist, _ = agent.actor(state_tensor)

            action = actor_dist.rsample()
            log_prob_action = actor_dist.log_prob(action)

            action_np = action.cpu().view(gym_env.action_space.shape).numpy()

            next_state, reward, terminated, truncated, _ = gym_env.step(action_np)

            step_count += 1
            
            done = terminated | truncated
            
            accum_episode_reward += reward

            reward_scaled = reward * reward_scaling

            episode.append(
                Memory(
                    state_tensor,
                    action,
                    log_prob_action,
                    reward_scaled,
                    done,
                    critic_value,
                )
            )

            state = next_state

            if divisible_by(step_count, steps_before_update) and len(episodes) > 0:
                actor_loss, critic_loss = agent.learn(episodes, aux_episodes)
                episodes.clear()
                num_policy_updates += 1
                
                if divisible_by(num_policy_updates, num_policy_iterations_before_aux) and mode is "ppg":
                    agent.learn_aux(aux_episodes)
                    aux_episodes.clear()
                    
                tqdm.write(
                    f"Mean Actor Loss: {actor_loss[-1]}, Mean Critic Loss: {critic_loss[-1]}, Mean Steps: {step_count}"
                )
                                
            if divisible_by(step_count, steps_before_eval):
                agent.save()

            if done:
                break
        
        tqdm.write(f"Accumulated Reward: {accum_episode_reward} in {n_steps} steps.")

        accum_episode_rewards.append(accum_episode_reward)

        episodes.append(episode)
