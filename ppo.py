import torch
from model import ShallowMLP
import gymnasium as gym
import random
from tqdm import tqdm

class PPO:
    
    def __init__(self, 
                 hidden_layer_dims=[32, 64, 32], 
                 p_dropout=0.2,
                 env = None,
                 lr_actor = 0.001,
                 lr_critic = 0.001,
                 ):
        
        self.env = env
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.shape[0]
        
        # Actor and Critic networks
        self.actor = ShallowMLP(self.observation_space, self.action_space, hidden_layer_dims, p_dropout)
        self.critic = ShallowMLP(self.observation_space, 1, hidden_layer_dims, p_dropout)
        
        # Initialize optimizer
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        
        # Initialize covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.action_space,), fill_value=0.5)
        self.cov_matrix = torch.diag(self.cov_var)
        
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 250            # timesteps per batch
        self.max_timesteps_per_episode = 250      # timesteps per episode
        self.gamma = 0.99                          # discount factor
        self.n_epochs = 1                       # number of epochs
    
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        
        # Mean of the Gaussian distribution
        mean = self.actor(state)
        
        # Covariance matrix
        cov_matrix = self.cov_matrix
        
        # Multivariate Gaussian distribution
        dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
        
        # Sample an action
        action = dist.sample()
        
        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)
                
        return action.detach().numpy(), log_prob.detach()
    
    def reset_env(self):
        
        state, _ = self.env.reset()
        
        for _ in range(random.randint(1, 30)):
            
            random_action = self.env.action_space.sample()
            
            state, reward, terminated, truncated, info = self.env.step(random_action)
            
        return state
    
    @staticmethod
    def get_discounted_return(batch_rewards: list[torch.FloatTensor], gamma: float):
    
        cum_returns: list[torch.FloatTensor] = []

        for rewards in batch_rewards:

            cum_return_list: list[float] = []
            cum_sum = 0
            
            for i, reward in enumerate(rewards.flip(0)):
                
                cum_sum = reward + gamma * cum_sum
                cum_return_list.append(cum_sum)

            cum_returns.append(torch.tensor(list(cum_return_list), dtype=torch.float32).flip(0))
                    
        return cum_returns
    
        
    def rollout(self):
        
        # Make some empty lists for logging
        batch_observations: list = []                # for observations
        batch_actions: list = []                     # for actions
        batch_log_probs: list = []                   # log probabilities
        batch_returns: list = []                     # for measuring episode returns
        batch_lens: list = []                        # for measuring episode lengths


        # Episodic data, keeps tracck of rewards per episode
        t = 0
        
        while t < self.timesteps_per_batch:
            
            state = self.reset_env()
            
            episodic_rewards: list = []
            episodic_length: int = 0
            
            # Remaining time in the batch
            for ep_t in range(self.max_timesteps_per_episode):
                
                # Tick / Tock
                t += 1
                
                if t >= self.timesteps_per_batch:
                    break

                batch_observations.append(state)
                
                # Sample from a normal distribution
                action, log_prob = self.get_action(state)
                                
                # Take a step in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Log observations, actions, and rewards
                batch_actions.append(action)
                episodic_rewards.append(reward)
                batch_log_probs.append(log_prob)
                
                state = next_state
                episodic_length += 1
                
                if terminated or truncated:
                    break
                
            batch_lens.append(ep_t + 1)
            batch_returns.append(torch.tensor(episodic_rewards, dtype=torch.float32))
            
        batch_observations = torch.tensor(batch_observations, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float32)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float32)
        
        batch_discounted_returns = self.get_discounted_return(batch_returns, self.gamma)
        batch_discounted_returns = torch.tensor(batch_discounted_returns, dtype=torch.float32)
        print(batch_discounted_returns.shape)

        self.env.close()
        
        return batch_observations, batch_actions, batch_log_probs, batch_discounted_returns, batch_lens
    
    def compute_advantage(self, batch_returns, batch_values):
        """
        Compute advantage using discounted returns and the value function.
        Advantage A(s, a) = R(s, a) - V(s).
        """
        batch_advantages = []
        
        for returns, values in zip(batch_returns, batch_values):
            
            advantages = returns - values
            
            batch_advantages.append(advantages)
            
        return batch_advantages
    
    def update_policy(self, observations, actions, log_probs, advantages):
        """
        Update the policy network using the PPO clipped objective.
        """
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        advantages = torch.stack(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            
            # Compute new log probabilities and the ratio
            mean = self.actor(observations)
            dist = torch.distributions.MultivariateNormal(mean, self.cov_matrix)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
            loss_actor = -torch.min(surrogate1, surrogate2).mean()

            # Update actor
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            
    def update_value_function(self, observations, returns):
        """
        Update the value function using the MSE loss.
        """
        observations = torch.stack(observations)
        returns = torch.stack(returns)

        for _ in range(self.n_epochs):
            values = self.critic(observations).squeeze()
            loss_critic = torch.nn.functional.mse_loss(values, returns)

            # Update critic
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            self.critic_optimizer.step()
    
    def train(self):
        
        for epoch in tqdm(range(self.n_epochs), desc="Training"):
        
            # 1) Collect trajectories
            batch_obs, batch_actions, batch_log_probs, batch_returns, batch_lens = self.rollout()
            
            # 2) Compute value estimates and advantages
            batch_values = self.critic(batch_obs).squeeze()
            
            
            
            advantages = self.compute_advantage(batch_returns, batch_values)
            print(advantages)
                        
            # # 3) Update actor and critic
            # self.update_policy(batch_obs_torch, batch_actions, batch_log_probs, advantages)
            # self.update_value_function(batch_obs_torch, discounted_returns)
            
            
    
env = gym.make("LunarLander-v3", continuous=True)

ppo = PPO(env=env)

ppo.train()

# The actor optimizes the policy using the advantage to decide how to adjust the probabilities of actions
# The critic optimizes the value function using the discounted returns as the target