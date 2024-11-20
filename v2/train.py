import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from vmas import make_env
from model import ActorNetwork, CriticNetwork

def train(
    env_name='v2',
    num_envs=100,
    n_agents=4,
    n_steps=32,
    total_timesteps=1000000,
    gamma=0.99,
    lam=0.95,
    lr=3e-5,
    clip_param=0.2,
    value_loss_coef=0.9,
    entropy_coef=0.001,
    max_grad_norm=0.05,
    epochs=10,
    minibatch_size=256,
    hidden_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    # Create environment with specified number of agents
    env = make_env(
        scenario=env_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        n_agents=n_agents,
    )

    # Get observation and action sizes
    obs_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].shape[0]

    # Initialize separate actor and critic models and optimizer
    actor_model = ActorNetwork(obs_size, action_size, hidden_size=hidden_size).to(device)
    critic_model = CriticNetwork(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(list(actor_model.parameters()) + list(critic_model.parameters()), lr=lr)

    # Track the best reward
    best_reward = -float('inf')

    # Storage for training data across steps
    obs = torch.zeros(n_steps, num_envs * n_agents, obs_size, device=device)
    actions = torch.zeros(n_steps, num_envs * n_agents, action_size, device=device)
    log_probs = torch.zeros(n_steps, num_envs * n_agents, device=device)
    rewards = torch.zeros(n_steps, num_envs * n_agents, device=device)
    dones = torch.zeros(n_steps, num_envs * n_agents, device=device)
    values = torch.zeros(n_steps, num_envs, device=device)

    # Initial observation
    current_obs = env.reset()
    current_obs = torch.cat(current_obs, dim=0).to(device)  # Shape: [num_envs * n_agents, obs_size]

    # Number of updates
    num_updates = total_timesteps // (n_steps * num_envs)

    for update in range(num_updates):
        for step in range(n_steps):
            obs[step].copy_(current_obs)

            # Forward pass: obtain decentralized policy and centralized value
            with torch.no_grad():
                dist = actor_model(current_obs)  # Action distribution, Shape: [num_envs * n_agents, action_size]

                # Get shared features for each agent
                agent_features = actor_model.actor_shared(current_obs)  # Shape: [num_envs *n_agents, hidden_size]

                # Reshape to [num_envs, n_agents, hidden_size] and aggregate
                agent_features = agent_features.view(num_envs, n_agents, hidden_size).mean(dim=1)  # Shape: [num_envs, hidden_size]

                # Debugging: Print shape to verify
                # print("Shape of aggregated agent_features for critic:", agent_features.shape)

                # Value prediction for each env
                centralized_value = critic_model(agent_features).squeeze(-1)  # Shape: [num_envs]

            actions_sample = dist.rsample()
            action_log_prob = dist.log_prob(actions_sample).sum(-1)

            # Clip actions to the valid range
            actions_sample = torch.clamp(actions_sample, -1.0, 1.0)

            actions[step].copy_(actions_sample)
            log_probs[step].copy_(action_log_prob)
            values[step].copy_(centralized_value)

            # Prepare actions for environment
            actions_env = actions_sample.view(num_envs, n_agents, action_size)  # Shape: [num_envs, n_agents, action_size]
            actions_list = [actions_env[:, i, :] for i in range(n_agents)]  # List of [num_envs, action_size]

            # Step environment and process rewards/dones
            obs_next, reward, done, info = env.step(actions_list)
            reward = torch.cat(reward, dim=0).to(device)  # Shape: [num_envs * n_agents,]
            done = torch.cat(done, dim=0).float().to(device) if isinstance(done, (list, tuple)) else done.float().to(device)  # Shape: [num_envs * n_agents,]

            if done.shape[0] == num_envs:
                done = done.repeat_interleave(n_agents)
            elif done.shape[0] != num_envs * n_agents:
                raise ValueError(f"Unexpected `done` shape: {done.shape}")

            rewards[step].copy_(reward)
            dones[step].copy_(done)

            # Prepare next observation
            current_obs = torch.cat(obs_next, dim=0).to(device)  # Shape: [num_envs * n_agents, obs_size]

        # Compute returns and advantages
        with torch.no_grad():
            # Get next value from the last step
            agent_features = actor_model.actor_shared(current_obs)  # Shape: [num_envs * n_agents, hidden_size]
            agent_features = agent_features.view(num_envs, n_agents, hidden_size).mean(dim=1)  # Shape: [num_envs, hidden_size]
            next_value = critic_model(agent_features).squeeze(-1)  # Shape: [num_envs]

        returns = torch.zeros_like(rewards, device=device)
        advantages = torch.zeros_like(rewards, device=device)

        gae = 0
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value.repeat_interleave(n_agents)  # Shape: [num_envs * n_agents]
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1].repeat_interleave(n_agents)  # Shape: [num_envs * n_agents]

            current_values = values[step].repeat_interleave(n_agents)  # Shape: [num_envs * n_agents]
            delta = rewards[step] + gamma * next_values * next_non_terminal - current_values
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = advantages[step] + current_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten rollout data
        b_obs = obs.reshape(-1, obs_size)                  # Shape: [n_steps * num_envs * n_agents, obs_size]
        b_actions = actions.reshape(-1, action_size)        # Shape: [n_steps * num_envs * n_agents, action_size]
        b_values = values.repeat_interleave(n_agents).view(-1)  # Shape: [n_steps * num_envs * n_agents]
        b_returns = returns.view(-1)                        # Shape: [n_steps * num_envs * n_agents]
        b_advantages = advantages.view(-1)                  # Shape: [n_steps * num_envs * n_agents]
        b_log_probs = log_probs.view(-1)                    # Shape: [n_steps * num_envs * n_agents]

        # PPO policy optimization
        for epoch in range(epochs):
            indices = np.arange(b_obs.size(0))
            np.random.shuffle(indices)

            for start in range(0, b_obs.size(0), minibatch_size):
                end = start + minibatch_size
                minibatch_indices = indices[start:end]

                batch_obs = b_obs[minibatch_indices]              # Shape: [minibatch_size, obs_size]
                batch_actions = b_actions[minibatch_indices]      # Shape: [minibatch_size, action_size]
                batch_values = b_values[minibatch_indices]        # Shape: [minibatch_size]
                batch_returns = b_returns[minibatch_indices]      # Shape: [minibatch_size]
                batch_advantages = b_advantages[minibatch_indices]  # Shape: [minibatch_size]
                batch_log_probs = b_log_probs[minibatch_indices]  # Shape: [minibatch_size]

                # Forward pass for action log probs and values
                dist = actor_model(batch_obs)  # Shape: [minibatch_size, action_size]
                agent_features = actor_model.actor_shared(batch_obs)  # Shape: [minibatch_size, hidden_size]

                # Ensure the batch size is divisible by n_agents
                if agent_features.size(0) % n_agents != 0:
                    raise ValueError("Batch size is not divisible by number of agents")

                batch_envs = agent_features.size(0) // n_agents  # e.g., 256 /4=64
                agent_features = agent_features.view(batch_envs, n_agents, hidden_size)  # Shape: [batch_envs, n_agents, hidden_size]

                # Aggregate agent features for critic
                aggregated_features = agent_features.mean(dim=1)  # Shape: [batch_envs, hidden_size]

                # Value prediction for each environment in the batch
                values_pred = critic_model(aggregated_features).squeeze(-1)  # Shape: [batch_envs]

                # Repeat values_pred for each agent to match batch size
                values_pred = values_pred.repeat_interleave(n_agents)  # Shape: [batch_envs * n_agents] = [minibatch_size]

                # Compute log probabilities and entropy
                action_log_probs = dist.log_prob(batch_actions).sum(-1)  # Shape: [minibatch_size]
                entropy = dist.entropy().sum(-1).mean()                 # Scalar

                # PPO objective
                ratios = torch.exp(action_log_probs - batch_log_probs)  # Shape: [minibatch_size]
                surr1 = ratios * batch_advantages                      # Shape: [minibatch_size]
                surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages  # Shape: [minibatch_size]
                action_loss = -torch.min(surr1, surr2).mean()          # Scalar

                # Value loss
                value_loss = F.mse_loss(values_pred, batch_returns)    # Shape: [minibatch_size]

                # Total loss
                loss = action_loss + value_loss_coef * value_loss - entropy_coef * entropy  # Scalar

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor_model.parameters()) + list(critic_model.parameters()), max_grad_norm)
                optimizer.step()

        # Logging and saving the best model
        total_reward = rewards.sum().item() / (num_envs * n_agents * n_steps)
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Total Reward per Agent: {total_reward:.2f}")

        # Save the model if it achieves a new best reward
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()}, 'ppo_feedforward_model_best.pth')
            print(f"New best model saved with reward: {best_reward:.2f}")

    # Final save of the model
    torch.save({'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()}, 'ppo_feedforward_model_final.pth')

if __name__ == '__main__':
    train()
