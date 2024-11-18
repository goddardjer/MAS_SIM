import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from vmas import make_env
from model import ActorCriticFeedforward


def train(
    env_name='wat',
    num_envs=1,
    n_agents=2,
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
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    # Create environment
    env = make_env(
        scenario=env_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        n_agents=n_agents,
    )

    # Manually set observation and action sizes
    obs_size = 18  # Set based on observed shape from Scenario's observation method
    action_size = 2  # Based on Action Space shape for each agent

    # Initialize the policy model and optimizer
    model = ActorCriticFeedforward(obs_size, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track the best reward
    best_reward = -float('inf')  # Initialize with a very low value

    # Storage for training data across steps
    obs = torch.zeros(n_steps, n_agents, obs_size).to(device)
    actions = torch.zeros(n_steps, n_agents, action_size).to(device)
    log_probs = torch.zeros(n_steps, n_agents).to(device)
    rewards = torch.zeros(n_steps, n_agents).to(device)
    dones = torch.zeros(n_steps, n_agents).to(device)
    values = torch.zeros(n_steps, n_agents).to(device)

    # Initial observation
    current_obs = env.reset()  # Shape: (n_agents, obs_size)

    num_updates = total_timesteps // (n_steps)

    for update in range(num_updates):
        for step in range(n_steps):
            obs[step].copy_(torch.stack(current_obs).view(n_agents, obs_size))

            # Forward pass through the model for each agent separately
            action_list, log_prob_list, value_list = [], [], []
            for i in range(n_agents):
                with torch.no_grad():
                    dist, value = model(current_obs[i].unsqueeze(0))  # Pass each agent's observation
                    action = dist.rsample()
                    action_log_prob = dist.log_prob(action).sum(-1)
                
                action = torch.clamp(action, -1.0, 1.0)
                
                action_list.append(action)
                log_prob_list.append(action_log_prob)
                value_list.append(value)

            # Stack the lists to create tensors
            actions_tensor = torch.cat(action_list).view(n_agents, action_size)
            log_probs_tensor = torch.cat(log_prob_list).view(n_agents)
            values_tensor = torch.cat(value_list).view(n_agents)

            actions[step].copy_(actions_tensor)
            log_probs[step].copy_(log_probs_tensor)
            values[step].copy_(values_tensor)

            # Prepare actions for environment
            actions_list = [action_list[i] for i in range(n_agents)]

            # Step environment and get new observations
            # Step environment and get new observations
            # Step environment and get new observations
            obs_next, reward, done, info = env.step(actions_list)

            # Ensure reward is a one-dimensional tensor before concatenation
            reward = torch.cat([r.unsqueeze(0) if r.dim() == 0 else r for r in reward], dim=0).to(device)

            # Check if `done` is a scalar or a list/tuple of tensors
            if isinstance(done, (list, tuple)):
                # Process each `done` value in the list
                done = torch.cat([d.unsqueeze(0) if d.dim() == 0 else d for d in done], dim=0).float().to(device)
            else:
                # If `done` is a single scalar tensor, expand it to match the number of agents
                done = done.expand(n_agents).float().to(device)


            # Copy adjusted `done` tensor
            rewards[step].copy_(reward.view(n_agents))
            dones[step].copy_(done.view(n_agents))


            # Prepare next observation
            current_obs = obs_next  # Do not concatenate, keep separate

        # Compute returns and advantages
        with torch.no_grad():
            next_value_list = [model(current_obs[i].unsqueeze(0))[1] for i in range(n_agents)]
            next_value = torch.cat(next_value_list).squeeze().view(n_agents)

        returns = torch.zeros_like(rewards).to(device)
        advantages = torch.zeros_like(rewards).to(device)

        gae = torch.zeros(n_agents).to(device)
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1]
            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = advantages[step] + values[step]

        # Flatten rollout data
        b_obs = obs.reshape(-1, obs_size)
        b_actions = actions.reshape(-1, action_size)
        b_values = values.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_log_probs = log_probs.reshape(-1)

        # PPO policy optimization
        for epoch in range(epochs):
            indices = np.arange(b_obs.size(0))
            np.random.shuffle(indices)

            for start in range(0, b_obs.size(0), minibatch_size):
                end = start + minibatch_size
                minibatch_indices = indices[start:end]

                batch_obs = b_obs[minibatch_indices]
                batch_actions = b_actions[minibatch_indices]
                batch_values = b_values[minibatch_indices]
                batch_returns = b_returns[minibatch_indices]
                batch_advantages = b_advantages[minibatch_indices]
                batch_log_probs = b_log_probs[minibatch_indices]

                # Forward pass for action log probs and values
                dist, values_pred = model(batch_obs)
                action_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # PPO objective: Calculate the ratio
                ratios = torch.exp(action_log_probs - batch_log_probs)

                # Clipped surrogate objective
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                action_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values_pred.squeeze(), batch_returns)

                # Total loss
                loss = action_loss + value_loss_coef * value_loss - entropy_coef * entropy

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Logging and saving the best model
        total_reward = rewards.sum().item() / (n_agents * n_steps)
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Total Reward per Agent: {total_reward:.2f}")

        # Save the model if it achieves a new best reward
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), 'ppo_feedforward_model_best.pth')
            print(f"New best model saved with reward: {best_reward:.2f}")

    # Final save of the model
    torch.save(model.state_dict(), 'ppo_feedforward_model_final.pth')


if __name__ == '__main__':
    train()
