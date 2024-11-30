# model.py

import torch
import torch.nn as nn
import torch.distributions as D

class Actor(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=256):
        super(Actor, self).__init__()

        # Actor network (decentralized execution)
        self.actor_network = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        # x: [batch_size, obs_size]
        features = self.actor_network(x)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)
        action_dist = D.Normal(mean, std)
        return action_dist, features  # Return action distribution and features

class Critic(nn.Module):
    def __init__(self, hidden_size=256):
        super(Critic, self).__init__()

        # Attention mechanism for agent interactions
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

        # Critic network
        self.critic_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, agent_features, num_agents):
        """
        agent_features: [num_envs * num_agents, hidden_size]
        num_agents: int
        """
        # Reshape to [num_agents, num_envs, hidden_size]
        features = agent_features.view(num_agents, -1, agent_features.size(-1))

        # Apply attention mechanism
        attn_output, _ = self.attention(features, features, features)

        # Aggregate attention outputs (mean over agents)
        attn_output = attn_output.mean(dim=0)  # Shape: [num_envs, hidden_size]

        # Compute value estimates
        values = self.critic_network(attn_output).squeeze(-1)  # Shape: [num_envs]

        return values
