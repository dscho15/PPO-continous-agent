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
        self.action_space = self.env.action_space.n
        
        # Actor and Critic networks
        self.actor = LinearAgent(self.observation_space, self.action_space, hidden_layer_dims, p_dropout)
        self.critic = LinearAgent(self.observation_space, 1, hidden_layer_dims, p_dropout)
        
        # Initialize optimizer
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(), lr=lr_critic)
        
        # Initialize covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(self.action_space,), fill_value=0.5)
        self.cov_matrix = torch.diag(self.cov_var)
    
    
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
                
        return action.detach(), log_prob.detach()
        
    def rollout(self):
        
        # make some empty lists for logging
        batch_obs: list = []            # for observations
        batch_acts: list = []           # for actions
        batch_weights: list = []        # for importance sampling weights
        batch_returns: list = []        # for measuring episode returns
        batch_lens: list = []           # for measuring episode lengths
        
        # while True:
            
        state, info = env.reset()
        done = False
        
        while not done:
            
            # sample from a normal distribution
            action, log_prob = self.action(state)
            
            print(self.env.step)
            print(action.shape)
            
            # take a step in the environment
            next_state, reward, terminated, truncated, info = self.env.step(action.tolist())
            
            # doesnt matter which one is true
            done = terminated | truncated
            
            # 
            batch_obs.append(state)
            batch_acts.append(action)

            state = next_state
                
        env.close()
        
        
    
    
    def learn(self):
        
        pass
        
        
        
if __name__ == "__main__":
    
    env = gym.make("LunarLander-v3")
    
    ppo = PPO(env=env)
    
    ppo.rollout()
    
    print(ppo.actor)
    
    ppo.learn()
    
    pass