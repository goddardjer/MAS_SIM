import torch
import torch.nn as nn
import torch.distributions as D

class SharedActorCritic(nn.Module):
    def __init__(self, obs_size, action_size, n_agents, hidden_size=256):
        super(SharedActorCritic, self).__init__()
        
        self.n_agents = n_agents
        
        # Actor shared layers (decentralized)
        self.actor_shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))
        
        # Critic shared layers (centralized)
        self.critic_shared = nn.Sequential(
            nn.Linear(obs_size * n_agents, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward_actor(self, x):
        """
        Forward pass for actor (decentralized).
        Args:
            x (Tensor): Individual agent observations.
        Returns:
            action_dist (Distribution): Action distribution for each agent.
        """
        x = self.actor_shared(x)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        action_dist = D.Normal(mean, std)
        return action_dist
    
    def forward_critic(self, x_all_agents):
        """
        Forward pass for critic (centralized).
        Args:
            x_all_agents (Tensor): Concatenated observations of all agents per environment.
        Returns:
            value (Tensor): Value estimates for each environment.
        """
        x = self.critic_shared(x_all_agents)
        value = self.critic(x).squeeze(-1)
        return value
