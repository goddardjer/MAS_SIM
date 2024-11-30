# train_block_pushing.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from vmas import make_env
from model import Actor, Critic

def train(
    env_name='v4',  # Path to your scenario file
    num_envs=128,
    n_steps=64,
    total_timesteps=1_000_000,
    gamma=0.99,
    lam=0.95,
    lr=3e-4,
    clip_param=0.2,
    value_loss_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    epochs=10,
    minibatch_size=2048,
    hidden_size=256,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    num_agents=2,  # Number of agents during training
):
    # Initialize wandb
    wandb.init(
        project="multi-agent-block-pushing",
        config={
            "env_name": env_name,
            "num_envs": num_envs,
            "n_steps": n_steps,
            "total_timesteps": total_timesteps,
            "gamma": gamma,
            "lam": lam,
            "lr": lr,
            "clip_param": clip_param,
            "value_loss_coef": value_loss_coef,
            "entropy_coef": entropy_coef,
            "max_grad_norm": max_grad_norm,
            "epochs": epochs,
            "minibatch_size": minibatch_size,
            "hidden_size": hidden_size,
            "device": device,
            "num_agents": num_agents,
        }
    )

    # Create environment with specified number of agents
    env = make_env(
        scenario=env_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
        n_agents=num_agents,
    )

    # Get observation and action sizes
    obs_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].shape[0]

    # Initialize actor and critic models
    actor_model = Actor(obs_size, action_size, hidden_size=hidden_size).to(device)
    critic_model = Critic(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(
        list(actor_model.parameters()) + list(critic_model.parameters()), lr=lr
    )

    # Watch models with wandb
    wandb.watch(actor_model, log="all")
    wandb.watch(critic_model, log="all")

    # Number of updates
    num_updates = total_timesteps // (n_steps * num_envs)

    # Initial observation
    current_obs = env.reset()
    num_agents = len(current_obs)
    num_envs = current_obs[0].shape[0]  # Updated num_envs in case it changes
    agent_obs = [obs.to(device) for obs in current_obs]

    for update in range(num_updates):
        # Storage for training data
        obs_buffer = []
        actions_buffer = []
        log_probs_buffer = []
        rewards_buffer = []
        dones_buffer = []
        agent_features_buffer = []

        for step in range(n_steps):
            # Concatenate observations from all agents
            obs_batch = torch.cat(agent_obs, dim=0)  # Shape: [num_envs * num_agents, obs_size]

            # Forward pass: obtain action distributions and agent features
            with torch.no_grad():
                action_dist, agent_features = actor_model(obs_batch)

            # Sample actions
            actions_sample = action_dist.sample()  # Shape: [num_envs * num_agents, action_size]
            action_log_probs = action_dist.log_prob(actions_sample).sum(-1)  # Shape: [num_envs * num_agents]

            # Prepare actions for environment
            actions_env = []
            for i in range(num_agents):
                idx = i * num_envs
                actions_env.append(actions_sample[idx:idx + num_envs])

            # Step environment
            obs_next, rewards, dones, infos = env.step(actions_env)

            # Collect data for rollout
            obs_buffer.append(obs_batch)
            actions_buffer.append(actions_sample)
            log_probs_buffer.append(action_log_probs)
            rewards_buffer.append(rewards)  # List of tensors per agent
            dones_buffer.append(dones)      # List of tensors per agent
            agent_features_buffer.append(agent_features)

            # Prepare next observation
            agent_obs = [obs.to(device) for obs in obs_next]

        # Flatten buffers
        obs_buffer = torch.stack(obs_buffer)  # Shape: [n_steps, num_envs * num_agents, obs_size]
        actions_buffer = torch.stack(actions_buffer)  # Shape: [n_steps, num_envs * num_agents, action_size]
        log_probs_buffer = torch.stack(log_probs_buffer)  # Shape: [n_steps, num_envs * num_agents]
        agent_features_buffer = torch.stack(agent_features_buffer)  # Shape: [n_steps, num_envs * num_agents, hidden_size]

        # Convert rewards and dones to tensors
        rewards_tensor = torch.zeros(n_steps, num_envs * num_agents, device=device)
        dones_tensor = torch.zeros(n_steps, num_envs * num_agents, device=device)
        for t in range(n_steps):
            for i in range(num_agents):
                idx = i * num_envs
                rewards_tensor[t, idx:idx + num_envs] = rewards_buffer[t][i].to(device)
                dones_tensor[t, idx:idx + num_envs] = dones_buffer[t][i].to(device)

        # Compute returns and advantages using GAE
        with torch.no_grad():
            # Compute values
            agent_features_flat = agent_features_buffer.view(n_steps * num_agents * num_envs, -1)
            values = critic_model(agent_features_flat, num_agents)  # Shape: [n_steps * num_envs]
            values = values.view(n_steps, num_envs).repeat(1, num_agents).view(n_steps, num_envs * num_agents)

            # Append one more step for bootstrap value
            next_values = torch.zeros_like(values[-1])
            advantages = torch.zeros_like(rewards_tensor)
            returns = torch.zeros_like(rewards_tensor)
            lastgaelam = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    nextnonterminal = 1.0 - dones_tensor[t]
                    nextvalues = next_values
                else:
                    nextnonterminal = 1.0 - dones_tensor[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards_tensor[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten buffers for PPO update
        obs_batch = obs_buffer.reshape(-1, obs_size)  # Shape: [n_steps * num_envs * num_agents, obs_size]
        actions_batch = actions_buffer.reshape(-1, action_size)
        log_probs_batch = log_probs_buffer.reshape(-1)
        returns_batch = returns.reshape(-1)
        advantages_batch = advantages.reshape(-1)

        # Normalize advantages
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        # Prepare dataset for DataLoader
        dataset = torch.utils.data.TensorDataset(
            obs_batch,
            actions_batch,
            log_probs_batch,
            returns_batch,
            advantages_batch,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=minibatch_size,
            shuffle=True,
        )

        # PPO policy optimization
        total_value_loss = 0
        total_policy_loss = 0
        total_entropy = 0
        num_updates_per_epoch = 0

        for epoch in range(epochs):
            for batch in dataloader:
                batch_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = [b.to(device) for b in batch]

                # Forward pass
                action_dist, agent_features = actor_model(batch_obs)

                # Compute action log probabilities and entropy
                action_log_probs = action_dist.log_prob(batch_actions).sum(-1)
                entropy = action_dist.entropy().sum(-1).mean()

                # Compute values using the critic
                values = critic_model(agent_features, num_agents)  # Shape: [batch_size]

                # Compute policy loss
                ratios = torch.exp(action_log_probs - batch_old_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute value loss
                value_loss = F.mse_loss(values, batch_returns)

                # Total loss
                loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor_model.parameters()) + list(critic_model.parameters()),
                    max_grad_norm,
                )
                optimizer.step()

                # Accumulate losses for logging
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
                total_entropy += entropy.item()
                num_updates_per_epoch += 1

        # Calculate average losses
        avg_value_loss = total_value_loss / num_updates_per_epoch
        avg_policy_loss = total_policy_loss / num_updates_per_epoch
        avg_entropy = total_entropy / num_updates_per_epoch

        # Logging metrics to wandb
        mean_reward = rewards_tensor.mean().item()
        wandb.log({
            'update': update,
            'mean_reward': mean_reward,
            'value_loss': avg_value_loss,
            'policy_loss': avg_policy_loss,
            'entropy': avg_entropy,
            'learning_rate': lr,
        })

        # Print logging information
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Mean Reward: {mean_reward:.2f}")

    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    train()
