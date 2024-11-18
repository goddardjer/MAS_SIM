# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class TanhNormal(torch.distributions.TransformedDistribution):
    def __init__(self, loc, scale):
        # Add a small epsilon to `scale` to prevent NaNs
        scale = scale.clamp(min=1e-6)
        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [torch.distributions.transforms.TanhTransform(cache_size=1)]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        return torch.tanh(self.base_dist.mean)

    def log_prob(self, value):
        # Inverse tanh for the value
        inverse_value = torch.atanh(value)
        log_prob = self.base_dist.log_prob(inverse_value) - torch.log(1 - value.pow(2) + 1e-6)
        return log_prob.sum(-1)

    def entropy(self):
        return self.base_dist.entropy()
    
class ActorCriticLSTM(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=128):
        super(ActorCriticLSTM, self).__init__()
        self.hidden_size = hidden_size

        # Shared layers
        self.fc1 = nn.Linear(obs_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)

        # Actor head
        self.actor_fc = nn.Linear(hidden_size, action_size)

        # Critic head
        self.critic_fc = nn.Linear(hidden_size, 1)

        # Action log std parameter (for continuous actions)
        self.log_std = nn.Parameter(torch.zeros(action_size))

    def forward(self, x, hx, cx):
        x = F.relu(self.fc1(x))
        hx, cx = self.lstm(x, (hx, cx))

        # Add debugging to detect NaNs in `hx` and `cx`
        if torch.isnan(hx).any() or torch.isnan(cx).any():
            print("NaN detected in LSTM hidden or cell state.")

        # Actor
        mean = self.actor_fc(hx)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)

        # Debugging to detect NaNs in `mean` and `std`
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("NaN detected in actor output.")
            print("Mean:", mean)
            print("Std:", std)

        dist = TanhNormal(mean, std)

        # Critic
        value = self.critic_fc(hx)

        # Debugging to detect NaNs in `value`
        if torch.isnan(value).any():
            print("NaN detected in critic output.")
            print("Value:", value)

        return dist, value, hx, cx

    def init_hidden_states(self, batch_size, device):
        hx = torch.zeros(batch_size, self.hidden_size).to(device)
        cx = torch.zeros(batch_size, self.hidden_size).to(device)
        return hx, cx


# class ActorCriticFeedforward(nn.Module):
#     def __init__(self, obs_size, action_size, hidden_size=128):
#         super(ActorCriticFeedforward, self).__init__()
        
#         # Shared layers
#         self.shared = nn.Sequential(
#             nn.Linear(obs_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#         )
        
#         # Actor head
#         self.actor_mean = nn.Linear(hidden_size, action_size)
#         self.actor_log_std = nn.Parameter(torch.zeros(action_size))  # Log standard deviation for actions
        
#         # Critic head
#         self.critic = nn.Linear(hidden_size, 1)

#     def forward(self, x):
#         # Shared forward pass
#         x = self.shared(x)
        
#         # Actor head
#         mean = self.actor_mean(x)
#         std = torch.exp(self.actor_log_std).expand_as(mean)
#         dist = dist.Normal(mean, std)
        
#         # Critic head
#         value = self.critic(x).squeeze(-1)  # Output value as a scalar
        
#         return dist, value

# model.py

import torch
import torch.nn as nn
import torch.distributions as D  # Import torch.distributions as D to avoid conflicts

class ActorCriticFeedforward(nn.Module):
    def __init__(self, obs_size, action_size, hidden_size=256):
        super(ActorCriticFeedforward, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Actor head
        self.actor_mean = nn.Linear(hidden_size, action_size)
        self.actor_log_std = nn.Parameter(torch.zeros(action_size))  # Log standard deviation for actions
        
        # Critic head
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Shared forward pass
        x = self.shared(x)
        
        # Actor head
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_log_std).expand_as(mean)
        action_dist = D.Normal(mean, std)  # Renamed from `dist` to `action_dist`
        
        # Critic head
        value = self.critic(x).squeeze(-1)  # Output value as a scalar
        
        return action_dist, value



