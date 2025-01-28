from algorithms.models import Actor, Critic
from tqdm import tqdm
from algorithms.dataset import (
    ExperienceDataset,
    Memory,
    MemoryAux,
    ExperienceAuxDataset,
)

from algorithms.utils import (
    update_network,
    to_torch_tensor,
    get_gae_advantages,
    get_returns,
    get_cum_returns_exact,
)

from algorithms.losses import (
    ClipActorLoss,
    EntropyActorLoss,
    ClipCriticLoss,
    SpectralEntropyLoss,
    KLDivLoss,
)

from ema_pytorch import EMA
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from collections import deque
from torch.utils.data import Dataset

import gymnasium as gym
import numpy as np
import torch

import torch


def create_shuffled_dataloader(
    episodes: list[Memory] | list[MemoryAux],
    dataset_func: callable,
    batch_size: int = 64,
    dataset: ExperienceDataset | ExperienceAuxDataset = None,
) -> torch.utils.data.DataLoader:

    if dataset is None:
        dataset_func = dataset_func(episodes)
    else:
        dataset_func = dataset

    return torch.utils.data.DataLoader(
        dataset_func, batch_size=batch_size, shuffle=True, drop_last=True
    )


def tuple_to_tensors_and_device(
    batch: tuple, device: torch.device
) -> tuple[torch.FloatTensor]:
    return [torch.tensor(b, dtype=torch.float32).to(device) for b in batch]


def divisible_by(x: int, y: int) -> bool:
    return x % y == 0


def normalize(x: torch.FloatTensor) -> torch.FloatTensor:
    return (x - x.mean()) / (x.std() + 1e-5)


class PPG(object):

    def __init__(
        self,
        dim_action_space: int,
        dim_obs_space: int,
        actor_kl_beta: float = 0.001,
        cautious_factor: float = 0.01,
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
        n_epochs: int = 1,
        n_trajectories: int = 1000,
        regen_reg_rate: float = 1e-4,
        save_path: str = "./model.pth",
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

        self.spec_entropy_actor_loss = SpectralEntropyLoss(0.1)
        self.spec_entropy_critic_loss = SpectralEntropyLoss(0.1)
        self.kl_div_loss = KLDivLoss(0.25)

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
        self, dataloader: torch.utils.data.DataLoader, episodes_aux: list[Memory]
    ) -> tuple[list[float]]:
        actor_losses, critic_losses = [], []

        for _ in range(self.n_epochs):
            epoch_actor_loss, epoch_critic_loss = [], []

            for batch in dataloader:

                # Prepare data
                batch = tuple_to_tensors_and_device(batch, self.device)
                (
                    states,
                    actions,
                    old_actions_log_probs,
                    rewards,
                    _,
                    old_values,
                    advantages,
                    returns,
                ) = batch

                # Actor optimization
                actor_dist, _ = self.actor(states)
                new_actions_log_probs = actor_dist.log_prob(actions)

                actor_loss = (
                    self.clip_actor_loss(
                        old_actions_log_probs, 
                        new_actions_log_probs, 
                        advantages
                    )
                    + self.spec_entropy_actor_loss(self.actor)
                    + self.entropy_actor_loss(actor_dist)
                )
                update_network(
                    actor_loss, self.actor, self.opt_actor, self.clip_actor_grads
                )
                epoch_actor_loss.append(actor_loss.item())

                # Critic optimization
                new_values = self.critic(states)
                critic_loss = self.clip_critic_loss(
                    old_values, new_values, returns
                ).mean() + self.spec_entropy_critic_loss(self.critic)
                update_network(
                    critic_loss, self.critic, self.opt_critic, self.clip_critic_grads
                )
                epoch_critic_loss.append(critic_loss.item())

                # Store auxiliary data
                episodes_aux.append(MemoryAux(states, actions, old_values, returns))

            # Record epoch losses
            actor_losses.append(np.mean(epoch_actor_loss))
            critic_losses.append(np.mean(epoch_critic_loss))

        return actor_losses, critic_losses

    def learn_aux(self, dataloader: torch.utils.data.DataLoader):

        for n in range(self.n_epochs * 6):

            for batch in dataloader:

                (states, actions, returns, old_values, old_log_prob_action) = batch

                actor_dist_new, policy_values = self.actor(states)
                new_log_prob_action = actor_dist_new.log_prob(actions)
                loss = self.clip_critic_loss(
                    old_values, policy_values, returns
                ) + self.kl_div_loss(new_log_prob_action, old_log_prob_action)
                update_network(loss, self.actor, self.opt_actor, self.clip_actor_grads)

                values = self.critic(states)
                loss = self.clip_critic_loss(old_values, values, returns)
                update_network(
                    loss, self.critic, self.opt_critic, self.clip_critic_grads
                )


def training_loop(
    gym_env: gym.Env,
    agent: PPG,
    n_training_loops: int = 10000,
    max_steps: int = 3000,
    steps_before_update: int = 5000,
    seed: int = 42,
    batch_size: int = 64,
):
    episodes = deque()
    aux_episodes = deque()
    step_count = 1
    num_policy_updates = 0

    for i in tqdm(range(n_training_loops), desc="Training Loop", position=0):
        state, info = gym_env.reset(seed=seed)
        episode = []

        while True:

            state_tensor = torch.from_numpy(state).float().to(agent.device)

            with torch.inference_mode():
                critic_value = agent.ema_critic.forward_eval(state_tensor)
                actor_dist, _ = agent.ema_actor.forward_eval(state_tensor)
                action = actor_dist.rsample()
                log_prob_action = actor_dist.log_prob(action)

            action_np = action.cpu().view(gym_env.action_space.shape).numpy()
            next_state, reward, terminated, truncated, _ = gym_env.step(action_np)

            step_count += 1
            done = terminated | truncated

            episode.append(
                Memory(state, action, log_prob_action, reward, done, critic_value)
            )
            state = next_state

            if divisible_by(step_count, steps_before_update) and len(episodes) > 0:

                mean_returns = torch.mean(
                    torch.tensor(
                        [torch.mean(get_cum_returns_exact(e)) for e in episodes]
                    )
                )

                dl = create_shuffled_dataloader(episodes, ExperienceDataset, batch_size)

                episodes.clear()

                actor_loss, critic_loss = agent.learn(dl, aux_episodes)

                tqdm.write(
                    f"Mean Actor Loss: {actor_loss[-1]},   \
                        Mean Critic Loss: {critic_loss[-1]}, \
                        Mean Returns: {mean_returns.item()}, \
                        Mean Steps: {step_count}"
                )

                num_policy_updates += 1

                if divisible_by(num_policy_updates, 32):

                    dataset = ExperienceAuxDataset(aux_episodes)

                    actor_dist, _ = agent.ema_actor.forward_eval(dataset.states)

                    dataset.action_log_probs = actor_dist.log_prob(dataset.actions)

                    dl = create_shuffled_dataloader(None, None, batch_size, dataset)

                    aux_episodes.clear()

                    agent.learn_aux(dl)

            # save the network every 10000 steps
            if divisible_by(step_count, steps_before_update * 4):
                agent.save()

            if done:
                break

        episodes.append(episode)
