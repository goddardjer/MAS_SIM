import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from vmas import make_env
from model import Actor, Critic  # Assuming you separated the actor and critic

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
    use_wandb=True,  # Enable or disable wandb logging
):
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(project="multi-agent-ppo", name="multi_agent_training")

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

    # Initialize actor, critic models, and optimizers
    actor_model = Actor(obs_size, action_size, hidden_size).to(device)
    critic_model = Critic(hidden_size).to(device)
    actor_optimizer = optim.Adam(actor_model.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic_model.parameters(), lr=lr)

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
                dist = actor_model(current_obs)
                agent_features = actor_model.shared_layers(current_obs)  # Assuming actor_model has shared layers for critic
                agent_features = agent_features.view(num_envs, n_agents, hidden_size).mean(dim=1)  # Aggregation
                centralized_value = critic_model(agent_features).squeeze(-1)  # Value prediction for each env

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
            # Get next value from the last step
            agent_features = actor_model.shared_layers(current_obs)
            agent_features = agent_features.view(num_envs, n_agents, hidden_size).mean(dim=1)  # Shape: [num_envs, hidden_size]
            next_value = critic_model(agent_features).squeeze(-1)  # Shape: [num_envs]

        returns = torch.zeros_like(rewards, device=device)
        advantages = torch.zeros_like(rewards, device=device)

        gae = 0
        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_non_terminal = 1.0 - dones[step]
                next_values = next_value.repeat_interleave(n_agents)
            else:
                next_non_terminal = 1.0 - dones[step + 1]
                next_values = values[step + 1].repeat_interleave(n_agents)

            current_values = values[step].repeat_interleave(n_agents)
            delta = rewards[step] + gamma * next_values * next_non_terminal - current_values
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[step] = gae
            returns[step] = advantages[step] + current_values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten rollout data
        b_obs = obs.reshape(-1, obs_size)
        b_actions = actions.reshape(-1, action_size)
        b_values = values.repeat_interleave(n_agents).view(-1)
        b_returns = returns.view(-1)
        b_advantages = advantages.view(-1)
        b_log_probs = log_probs.view(-1)

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
                dist = actor_model(batch_obs)
                agent_features = actor_model.shared_layers(batch_obs)
                values_pred = critic_model(agent_features.view(-1, hidden_size)).squeeze(-1)

                action_log_probs = dist.log_prob(batch_actions).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                # PPO objective
                ratios = torch.exp(action_log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                action_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values_pred, batch_returns)

                # Total loss
                loss = action_loss + value_loss_coef * value_loss - entropy_coef * entropy

                # Backpropagation and optimization
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor_model.parameters(), max_grad_norm)
                nn.utils.clip_grad_norm_(critic_model.parameters(), max_grad_norm)
                actor_optimizer.step()
                critic_optimizer.step()

                # Log metrics to wandb if enabled
                if use_wandb:
                    wandb.log({
                        "loss/action_loss": action_loss.item(),
                        "loss/value_loss": value_loss.item(),
                        "loss/entropy": entropy.item(),
                        "reward/total_reward": rewards.sum().item() / (num_envs * n_agents * n_steps),
                        # Add specific components of the reward system if you want to track them
                    })

        # Save the model if it achieves a new best reward
        total_reward = rewards.sum().item() / (num_envs * n_agents * n_steps)
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Total Reward per Agent: {total_reward:.2f}")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()}, 'ppo_feedforward_model_best.pth')
            print(f"New best model saved with reward: {best_reward:.2f}")

    torch.save({'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()}, 'ppo_feedforward_model_final.pth')

if __name__ == '__main__':
    train(use_wandb=False)
