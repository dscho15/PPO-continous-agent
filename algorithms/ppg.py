from algorithms.models import Actor, Critic
from algorithms.dataset import ExperienceDataset, Memory, create_dataloader
from algorithms.utils import (
    optimize_network,
    to_torch_tensor,
    get_gae_advantages,
    get_cum_returns,
)
from algorithms.losses import (
    ClipActorLoss,
    EntropyActorLoss,
    ClipCriticLoss,
    SpectralEntropyLoss,
)

from collections import deque
from torch.utils.data import Dataset

import gymnasium as gym
import numpy as np
import torch

from tqdm import tqdm
import torch
from einops import reduce, einsum


def tuple_to_tensors_and_device(
    batch: tuple[torch.FloatTensor], device
) -> tuple[torch.FloatTensor]:
    return [torch.tensor(b, dtype=torch.float32).to(device) for b in batch]


class PPG(object):

    def __init__(
        self,
        dim_obs_space: int,
        dim_action_space: int,
        actor_kl_beta: float = 0.001,
        clip_actor_grads: float = 0.5,
        clip_critic_grads: float = 0.5,
        clip_actor_eps: float = 0.2,
        clip_critic_eps: float = 0.5,
        device: str = "cuda:0",
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        n_critic_epochs: int = 8,
        n_policy_epochs: int = 4,
        n_trajectories: int = 1000,
    ):
        self.observation_space = dim_obs_space
        self.action_space = dim_action_space

        self.actor = Actor(self.observation_space, self.action_space)
        self.critic = Critic(self.observation_space)

        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)

        self.clip_actor_loss = ClipActorLoss(clip_actor_eps)
        self.entropy_actor_loss = EntropyActorLoss(actor_kl_beta)
        self.clip_critic_loss = ClipCriticLoss(clip_critic_eps)
        self.spec_entropy_actor_loss = SpectralEntropyLoss()
        self.spec_entropy_critic_loss = SpectralEntropyLoss()

        self.clip_actor_grads = clip_actor_grads
        self.clip_critic_grads = clip_critic_grads
        self.clip_eps = clip_actor_eps
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_critic_epochs = n_critic_epochs
        self.n_policy_epochs = n_policy_epochs
        self.n_trajectories = n_trajectories

        self.actor.to(self.device)
        self.critic.to(self.device)

    def mode(self, mode: str):
        modes = {
            "train": lambda: (self.actor.train(), self.critic.train()),
            "eval": lambda: (self.actor.eval(), self.critic.eval()),
        }
        if mode in modes:
            modes[mode]()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'eval'.")

    def learn(self, dataloader: torch.utils.data.DataLoader):

        actor_losses = []

        for _ in range(self.n_policy_epochs):
            
            loss = []

            for batch in dataloader:

                batch = tuple_to_tensors_and_device(batch, self.device)

                (
                    states,
                    actions,
                    old_log_prob_actions,
                    _,
                    _,
                    _,
                    advantages,
                    _,
                ) = batch

                if advantages.shape != (self.batch_size, 1):
                    advantages = advantages.unsqueeze(1)

                dist, _ = self.actor(states)

                new_log_prob_actions = dist.log_prob(actions)

                actor_loss = (
                    self.clip_actor_loss(
                        old_log_prob_actions, new_log_prob_actions, advantages
                    )
                    + self.spec_entropy_actor_loss(self.actor)
                    + self.entropy_actor_loss(dist)
                )

                optimize_network(
                    actor_loss,
                    self.actor,
                    self.actor_optimizer,
                    self.clip_actor_grads,
                )
                
                loss.append(actor_loss.item())
                
            actor_losses.append(np.mean(loss))
            
        # Critic training
            
        critic_losses = []

        for _ in range(self.n_critic_epochs):
            
            loss = []

            for batch in dataloader:

                batch = tuple_to_tensors_and_device(batch, self.device)

                (
                    obs,
                    _,
                    _,
                    _,
                    _,
                    old_values,
                    _,
                    returns,
                ) = batch

                new_values = self.critic(obs)

                critic_loss = self.clip_critic_loss(
                    old_values, new_values, returns
                ).mean()
                critic_loss += self.spec_entropy_critic_loss(self.critic)

                optimize_network(
                    critic_loss,
                    self.critic,
                    self.critic_optimizer,
                    self.clip_critic_grads,
                )
                
                loss.append(critic_loss.item())

            critic_losses.append(np.mean(loss))
            
        return actor_losses, critic_losses


def training_loop(
    gym: gym.Env,
    agent: PPG,
    n_max_que_size: int = 20,
    n_training_loops: int = 1000,
    n_max_number_of_steps: int = 1000,
    n_steps_before_update: int = 5000,
    seed: int = 42,
    batch_size: int = 64,
):
    episodes = deque(maxlen=20)
    n_steps = 1
    
    loop_iterator = tqdm(range(n_training_loops), desc="Training Loop", position=0)

    for i in loop_iterator:

        state, info = gym.reset(seed=seed)
        episode = []
        
        for t in range(n_max_number_of_steps):

            with torch.inference_mode():

                state = to_torch_tensor(state, agent.device)

                value = agent.critic(state)
                                
                dist, _ = agent.actor(state)

                a = dist.sample()

                log_a = dist.log_prob(a)

            s_next, reward, terminated, truncated, _ = gym.step(
                a.cpu().view(gym.action_space.shape).numpy()
            )

            n_steps += 1

            done = terminated | truncated

            mem = Memory(state, a, log_a, reward, done, value)
            episode.append(mem)

            state = s_next

            if n_steps % n_steps_before_update == 0:
                                
                episodes_popped = [episodes.pop() for _ in range(len(episodes))]
                episodes_popped = [e for e in episodes_popped if len(e) > 0]
                
                if len(episodes_popped) == 0:
                    assert len(episodes) == 0, "Episodes should be empty"
                    
                advantages = [get_gae_advantages(e) for e in episodes_popped]
                
                mean_disc_cum_rewards = torch.tensor([torch.mean(get_cum_returns(e)) for e in episodes_popped]).mean()
                
                dataloader = create_dataloader(episodes_popped, advantages, batch_size)
                
                actor_losses, critic_losses = agent.learn(dataloader)
                
                loop_iterator.set_postfix(
                    actor_loss=actor_losses[-1],
                    critic_loss=critic_losses[-1],
                    exp_return=mean_disc_cum_rewards.item(),
                )
                
                episodes = []
                
            if done:
                break

        episodes.append(episode)
