import torch
import torch.nn as nn
import torch.distributions as D

class Actor(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=256):
        super(Actor, self).__init__()
        
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
    
    def forward(self, x):
        """
        Forward pass for actor.
        Args:
            x (Tensor): Individual agent observations. Shape: [batch_size, obs_size]
        Returns:
            action_dist (Distribution): Action distribution for each agent.
        """
        x = self.actor_shared(x)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        action_dist = D.Normal(mean, std)
        return action_dist


class Critic(nn.Module):
    def __init__(self, hidden_size=256):
        super(Critic, self).__init__()
        
        # Critic shared layers
        self.critic_shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, aggregated_features):
        """
        Forward pass for critic (centralized).
        Args:
            aggregated_features (Tensor): Aggregated agent features. Shape: [batch_size, hidden_size]
        Returns:
            value (Tensor): Value estimates for each environment. Shape: [batch_size]
        """
        x = self.critic_shared(aggregated_features)  # Shape: [batch_size, hidden_size]
        value = self.critic(x).squeeze(-1)          # Shape: [batch_size]
        return value


# class SharedActorCritic(nn.Module):
#     def __init__(self, obs_size, action_size, hidden_size=256):
#         super(SharedActorCritic, self).__init__()
        
#         # Separate actor and critic networks
#         self.actor = ActorNetwork(obs_size, action_size, hidden_size)
#         self.critic = CriticNetwork(hidden_size)
    
#     def forward_actor(self, x):
#         """
#         Forward pass for actor network.
#         Args:
#             x (Tensor): Individual agent observations.
#         Returns:
#             action_dist (Distribution): Action distribution for each agent.
#         """
#         return self.actor(x)
    
#     def forward_critic(self, agent_features):
#         """
#         Forward pass for critic network.
#         Args:
#             agent_features (Tensor): Aggregated agent features. Shape: [batch_size, hidden_size]
#         Returns:
#             value (Tensor): Value estimates for each environment.
#         """
#         return self.critic(agent_features)
