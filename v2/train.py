import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from vmas import make_env
from model import SharedActorCritic

def train(
    env_name='v2',
    num_envs=10,
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

    # Initialize model and optimizer
    model = SharedActorCritic(obs_size, action_size, n_agents, hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track the best reward
    best_reward = -float('inf')

    # Storage for training data across steps
    obs = torch.zeros(n_steps, num_envs * n_agents, obs_size, device=device)
    actions = torch.zeros(n_steps, num_envs * n_agents, action_size, device=device)
    log_probs = torch.zeros(n_steps, num_envs * n_agents, device=device)
    rewards = torch.zeros(n_steps, num_envs * n_agents, device=device)
    dones = torch.zeros(n_steps, num_envs * n_agents, device=device)
    values = torch.zeros(n_steps, num_envs, device=device)  # Centralized value per environment

    # Initial observation
    current_obs = env.reset()
    current_obs = torch.cat(current_obs, dim=0).to(device)

    # Number of updates
    num_updates = total_timesteps // (n_steps * num_envs)

    for update in range(num_updates):
        for step in range(n_steps):
            obs[step].copy_(current_obs)

            # Forward pass: obtain decentralized policy and centralized value
            with torch.no_grad():
                dist = model.forward_actor(current_obs)
                # Use centralized critic with concatenated observations for each environment
                centralized_value = model.forward_critic(
                    current_obs.view(num_envs, n_agents * obs_size)
                ).squeeze(-1)  # Shape [num_envs]

            actions_sample = dist.rsample()
            action_log_prob = dist.log_prob(actions_sample).sum(-1)

            # Clip actions to the valid range
            actions_sample = torch.clamp(actions_sample, -1.0, 1.0)

            actions[step].copy_(actions_sample)
            log_probs[step].copy_(action_log_prob)
            values[step].copy_(centralized_value)  # Store centralized value for each environment

            # Prepare actions for environment
            actions_env = actions_sample.view(num_envs, n_agents, action_size)
            actions_list = [actions_env[:, i, :] for i in range(n_agents)]

            # Step environment and process rewards/dones
            obs_next, reward, done, info = env.step(actions_list)
            reward = torch.cat(reward, dim=0).to(device)
            done = torch.cat(done, dim=0).float().to(device) if isinstance(done, (list, tuple)) else done.float().to(device)

            if done.shape[0] == num_envs:
                done = done.repeat_interleave(n_agents)
            elif done.shape[0] != num_envs * n_agents:
                raise ValueError(f"Unexpected `done` shape: {done.shape}")

            rewards[step].copy_(reward)
            dones[step].copy_(done)

            # Prepare next observation
            current_obs = torch.cat(obs_next, dim=0).to(device)

        # Compute returns and advantages
        with torch.no_grad():
            next_value = model.forward_critic(
                current_obs.view(num_envs, n_agents * obs_size)
            ).squeeze(-1)  # Shape [num_envs]

        returns = torch.zeros_like(rewards, device=device)      # Shape [n_steps, num_envs * n_agents]
        advantages = torch.zeros_like(rewards, device=device)  # Shape [n_steps, num_envs * n_agents]

        gae = 0
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - dones[step]  # Shape [40]
                # Repeat next_value for each agent
                next_values = next_value.repeat_interleave(n_agents)  # Shape [40]
            else:
                next_non_terminal = 1.0 - dones[step + 1]  # Shape [40]
                next_values = values[step + 1].squeeze().repeat_interleave(n_agents)  # Shape [40]

            # Repeat current values for each agent
            current_values = values[step].squeeze().repeat_interleave(n_agents)  # Shape [40]

            delta = rewards[step] + gamma * next_values * next_non_terminal - current_values  # Shape [40]
            gae = delta + gamma * lam * next_non_terminal * gae  # Shape [40]
            advantages[step] = gae
            returns[step] = advantages[step] + current_values  # Shape [40]

        # Flatten rollout data
        b_obs = obs.reshape(-1, obs_size)                # Shape [n_steps * num_envs * n_agents, obs_size]
        b_actions = actions.reshape(-1, action_size)      # Shape [n_steps * num_envs * n_agents, action_size]
        b_values = values.view(-1).repeat_interleave(n_agents)  # Shape [n_steps * num_envs * n_agents]
        b_returns = returns.view(-1)                      # Shape [n_steps * num_envs * n_agents]
        b_advantages = advantages.view(-1)                # Shape [n_steps * num_envs * n_agents]
        b_log_probs = log_probs.view(-1)                  # Shape [n_steps * num_envs * n_agents]

        # PPO policy optimization
        for epoch in range(epochs):
            indices = np.arange(b_obs.size(0))
            np.random.shuffle(indices)

            for start in range(0, b_obs.size(0), minibatch_size):
                end = start + minibatch_size
                minibatch_indices = indices[start:end]

                # Ensure minibatch_indices are within bounds
                if end > b_returns.size(0):
                    minibatch_indices = indices[start:]
                    end = b_returns.size(0)

                batch_obs = b_obs[minibatch_indices]             # Shape [minibatch_size, obs_size]
                batch_actions = b_actions[minibatch_indices]     # Shape [minibatch_size, action_size]
                batch_values = b_values[minibatch_indices]       # Shape [minibatch_size]
                batch_returns = b_returns[minibatch_indices]     # Shape [minibatch_size]
                batch_advantages = b_advantages[minibatch_indices]  # Shape [minibatch_size]
                batch_log_probs = b_log_probs[minibatch_indices]    # Shape [minibatch_size]

                # Forward pass for action log probs and values
                dist = model.forward_actor(batch_obs)
                values_pred = model.forward_critic(
                    batch_obs.view(-1, n_agents * obs_size)
                ).squeeze()  # Shape [minibatch_size / n_agents]

                # Since centralized critic provides one value per environment, we need to repeat it per agent
                values_pred = values_pred.repeat_interleave(n_agents)  # Shape [minibatch_size]

                # Compute log probabilities and entropy
                action_log_probs = dist.log_prob(batch_actions).sum(-1)  # Shape [minibatch_size]
                entropy = dist.entropy().sum(-1).mean()                 # Scalar

                # PPO objective
                ratios = torch.exp(action_log_probs - batch_log_probs)  # Shape [minibatch_size]
                surr1 = ratios * batch_advantages                      # Shape [minibatch_size]
                surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages  # Shape [minibatch_size]
                action_loss = -torch.min(surr1, surr2).mean()          # Scalar

                # Value loss
                value_loss = F.mse_loss(values_pred, batch_returns)    # Scalar

                # Total loss
                loss = action_loss + value_loss_coef * value_loss - entropy_coef * entropy  # Scalar

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Logging and saving the best model
        total_reward = rewards.sum().item() / (num_envs * n_agents * n_steps)
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Total Reward per Agent: {total_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), 'ppo_feedforward_model_best.pth')
            print(f"New best model saved with reward: {best_reward:.2f}")

    # Final save of the model
    torch.save(model.state_dict(), 'ppo_feedforward_model_final.pth')

if __name__ == '__main__':
    train()
