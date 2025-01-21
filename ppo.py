import torch
from model import LinearAgent
import gymnasium as gym

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
        self.actor = LinearAgent(self.observation_space, self.action_space, hidden_layer_dims, p_dropout)
        self.critic = LinearAgent(self.observation_space, 1, hidden_layer_dims, p_dropout)
        
        # Initialize optimizer
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        
        # Initialize covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.action_space,), fill_value=0.5)
        self.cov_matrix = torch.diag(self.cov_var)
        
        self._init_hyperparameters()
        
    def _init_hyperparameters(self):
        
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 4800            # timesteps per batch
        self.max_timesteps_per_episode = 1600      # timesteps per episode
    
    
    def action(self, state):
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
        
    def rollout(self):
        
        # Make some empty lists for logging
        batch_observations: list = []                # for observations
        batch_actions: list = []                     # for actions
        batch_log_probs: list = []                   # log probabilities
        batch_weights: list = []                     # for importance sampling weights
        batch_returns: list = []                     # for measuring episode returns
        batch_lens: list = []                        # for measuring episode lengths


        # Episodic data, keeps tracck of rewards per episode
        t = 0
        
        while t < self.timesteps_per_batch:
            
            observation, _ = self.env.reset()
            done: bool = False
            
            episodic_rewards: list = []
            episodic_length: int = 0
            
            # Remaining time in the batch
            for ep_t in range(self.max_timesteps_per_episode):

                if self.render:
                    env.render()
                
                # Tick / Tock
                t += 1

                batch_observations.append(observation)
                
                # Sample from a normal distribution
                action, log_prob = self.action(state)
                                
                # Take a step in the environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Log observations, actions, and rewards
                batch_actions.append(action)
                episode_rewards.append(reward)
                batch_log_probs.append(log_prob)
                
                if t == self.timesteps_per_batch or terminated or truncated:
                    break
                
                state = next_state
                episode_length += 1

            batch_lens.append(ep_t + 1)
            batch

        self.env.close()
        
        return batch_observations, batch_actions, batch_weights, batch_returns, batch_lens

        
        
    
    
    def learn(self):
        
        pass
        
        
        
if __name__ == "__main__":
    
    env = gym.make("LunarLander-v3", continuous=True)
    
    ppo = PPO(env=env)
    
    ppo.rollout()
    
    print(ppo.actor)
    
    ppo.learn()
    
    pass
