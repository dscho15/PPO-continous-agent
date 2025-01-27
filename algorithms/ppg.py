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

from ema_pytorch import EMA
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

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

def divisible_by(x: int, y: int) -> bool:
    return x % y == 0


class PPG(object):

    def __init__(
        self,
        dim_action_space: int,
        dim_obs_space: int,
        actor_kl_beta: float = 0.001,
        cautious_factor: float = 0.01,
        clip_actor_eps: float = 0.2,
        clip_actor_grads: float = 0.5,
        clip_critic_eps: float = 0.5,
        clip_critic_grads: float = 0.5,
        device: str = "cuda:0",
        ema_decay: float = 0.90,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-4,
        n_epochs: int = 4,
        n_trajectories: int = 1000,
        regen_reg_rate: float = 1e-4,
        save_path: str = "./model.pth"
    ):
        self.observation_space = dim_obs_space
        self.action_space = dim_action_space
        self.save_path = save_path

        self.actor = Actor(self.observation_space, self.action_space).to(device)
        self.critic = Critic(self.observation_space).to(device)

        self.ema_actor = EMA(
            self.actor,
            beta=ema_decay,
            include_online_model=False,
            update_model_with_ema_every=1000,
        )
        self.ema_critic = EMA(
            self.critic,
            beta=ema_decay,
            include_online_model=False,
            update_model_with_ema_every=1000,
        )

        self.opt_actor = AdoptAtan2(
            self.actor.parameters(),
            lr=lr_actor,
            cautious_factor=cautious_factor,
            regen_reg_rate=regen_reg_rate,
        )
        self.opt_critic = AdoptAtan2(
            self.critic.parameters(),
            lr=lr_critic,
            cautious_factor=cautious_factor,
            regen_reg_rate=regen_reg_rate,
        )
        
        self.ema_actor.add_to_optimizer_post_step_hook(self.opt_actor)
        self.ema_critic.add_to_optimizer_post_step_hook(self.opt_critic)

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
        self.n_epochs = n_epochs
        self.n_trajectories = n_trajectories

        self.actor.to(self.device)
        self.critic.to(self.device)
        
    def save(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, str(self.save_path))
        
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

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
        critic_losses = []

        for _ in range(self.n_epochs):
            
            a_loss = []
            c_loss = []
            
            for batch in dataloader:

                batch = tuple_to_tensors_and_device(batch, self.device)

                (
                    states,
                    actions,
                    old_actions_log_probs,
                    reward,
                    done,
                    old_values,
                    advantages,
                    returns,
                ) = batch

                advantages = advantages.unsqueeze(1)                
                
                # Actor loss
                dist, _ = self.actor(states)
                new_actions_log_probs = dist.log_prob(actions)
                
                loss = (
                    self.clip_actor_loss(
                        old_actions_log_probs, new_actions_log_probs, advantages
                    )
                    + self.spec_entropy_actor_loss(self.actor)
                    + self.entropy_actor_loss(dist)
                )
                
                a_loss.append(loss.item())
                
                optimize_network(
                    loss,
                    self.actor,
                    self.opt_actor,
                    self.clip_actor_grads,
                )
                
                # Critic loss
                new_values = self.critic(states)
                
                loss = (
                    self.clip_critic_loss(old_values, new_values, returns).mean() +
                    self.spec_entropy_critic_loss(self.critic)
                )

                optimize_network(
                    loss,
                    self.critic,
                    self.opt_critic,
                    self.clip_critic_grads,
                )
                
                c_loss.append(loss.item())

            actor_losses.append(np.mean(a_loss))
            critic_losses.append(np.mean(c_loss))

        return actor_losses, critic_losses


def training_loop(
    gym: gym.Env,
    agent: PPG,
    n_max_que_size: int = 50,
    n_training_loops: int = 1000,
    n_max_number_of_steps: int = 1000,
    n_steps_before_update: int = 5000,
    seed: int = 42,
    batch_size: int = 64
):
    episodes = deque(maxlen=n_max_que_size)
    n_steps = 1
    
    loop_iterator = tqdm(range(n_training_loops), desc="Training Loop", position=0)

    for i in loop_iterator:

        state, info = gym.reset(seed=seed)
        episode = []

        for t in range(n_max_number_of_steps):

            state = torch.from_numpy(state).float().to(agent.device)
            
            value = agent.ema_critic.forward_eval(state)
            
            dist, _ = agent.ema_actor.forward_eval(state)
            
            a = dist.sample()
            log_a = dist.log_prob(a)

            next_state, reward, terminated, truncated, _ = gym.step(
                a.cpu().view(gym.action_space.shape).numpy()
            )

            n_steps += 1

            done = terminated | truncated

            mem = Memory(state, a, log_a, reward, done, value)
            
            episode.append(mem)

            state = next_state

            if divisible_by(n_steps, n_steps_before_update):

                episodes_popped = [episodes.pop() for _ in range(len(episodes))]
                episodes_popped = [e for e in episodes_popped if len(e) > 0]

                if len(episodes_popped) == 0:
                    assert len(episodes) == 0, "Episodes should be empty"

                advantages = [get_gae_advantages(e) for e in episodes_popped]

                mean_disc_cum_rewards = torch.tensor(
                    [torch.mean(get_cum_returns(e)) for e in episodes_popped]
                ).mean()

                dataloader = create_dataloader(episodes_popped, advantages, batch_size)

                actor_losses, critic_losses = agent.learn(dataloader)

                loop_iterator.set_postfix(
                    actor_loss=actor_losses[-1],
                    critic_loss=critic_losses[-1],
                    exp_return=mean_disc_cum_rewards.item(),
                    n_steps=n_steps,
                )

                episodes = []

            if done:
                break

        episodes.append(episode)
