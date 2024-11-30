import torch
import torch.nn as nn
import torch.distributions as D


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=256):
        super(Actor, self).__init__()

        # Actor shared layers (decentralized execution)
        self.actor_shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x):
        """
        Forward pass for the actor network.

        Args:
            x (Tensor): Individual agent observations. Shape: [batch_size, obs_size]

        Returns:
            action_dist (Distribution): Action distribution for each agent.
            agent_features (Tensor): Features for each agent. Shape: [batch_size, hidden_size]
        """
        agent_features = self.actor_shared(x)
        mean = self.actor_mean(agent_features)
        std = torch.exp(self.actor_log_std)
        action_dist = D.Normal(mean, std)
        return action_dist, agent_features  # Return both the distribution and agent features


class Critic(nn.Module):
    def __init__(self, hidden_size=256):
        super(Critic, self).__init__()

        # Attention mechanism for agent interactions
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

        # Critic layers
        self.critic_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, agent_features_padded, agent_masks):
        """
        Forward pass for the critic network.

        Args:
            agent_features_padded (Tensor): Padded agent features.
                Shape: [max_agents, batch_size, hidden_size]
            agent_masks (Tensor): Mask indicating valid agent positions.
                Shape: [batch_size, max_agents]

        Returns:
            values (Tensor): Value estimates for each environment. Shape: [batch_size]
        """
        # Apply attention mechanism
        # agent_features_padded: [max_agents, batch_size, hidden_size]
        # agent_masks: [batch_size, max_agents]
        attn_output, _ = self.attention(
            agent_features_padded,
            agent_features_padded,
            agent_features_padded,
            key_padding_mask=~agent_masks  # Mask invalid (padded) positions
        )
        # Aggregate attention outputs
        attn_output = attn_output.mean(dim=0)  # Shape: [batch_size, hidden_size]
        # Pass through critic layers
        value = self.critic_layers(attn_output).squeeze(-1)  # Shape: [batch_size]
        return value
