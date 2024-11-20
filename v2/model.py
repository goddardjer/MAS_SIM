import torch
import torch.nn as nn
import torch.distributions as D

class SharedActorCritic(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=256):
        super(SharedActorCritic, self).__init__()
        
        # Shared feature extraction layers
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor head (decentralized)
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))  # Log standard deviation for actions
        
        # Centralized critic head
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, centralized=False):
        # Shared feature extraction
        x = self.shared(x)
        
        # Actor head (shared across all agents)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        action_dist = D.Normal(mean, std)
        
        # If centralized, use the centralized critic
        if centralized:
            # Assuming x includes all agents' observations concatenated together for centralized value
            value = self.critic(x).squeeze(-1)
        else:
            # Decentralized value (for individual agents)
            value = self.critic(x).squeeze(-1)

        return action_dist, value
