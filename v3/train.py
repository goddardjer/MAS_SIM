import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from vmas import make_env
from model import Actor, Critic


def train(
    env_name='v3',
    num_envs=128,
    n_steps=64,
    total_timesteps=10_000_000,
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
):
    # Initialize wandb
    wandb.init(
        project="multi-agent-ppo",
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
        }
    )

    # Create environment
    env = make_env(
        scenario=env_name,
        num_envs=num_envs,
        device=device,
        continuous_actions=True,
    )

    # Get observation and action sizes
    obs_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].shape[0]
    num_agents = env.num_agents  # Number of agents in each environment

    # Initialize actor and critic models
    actor_model = Actor(obs_size, action_size, hidden_size=hidden_size).to(device)
    critic_model = Critic(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(
        list(actor_model.parameters()) + list(critic_model.parameters()), lr=lr
    )

    # Watch models with wandb
    wandb.watch(actor_model, log="all")
    wandb.watch(critic_model, log="all")

    # Track the best average reward
    best_avg_reward = -float('inf')

    # Initial observation
    current_obs = env.reset()  # List of observations per environment
    # current_obs is a list of length num_envs, each element is a list of observations for each agent
    # Flatten observations and track agent indices
    agent_obs = []
    for env_obs in current_obs:
        for obs in env_obs:
            agent_obs.append(obs)
    agent_obs = torch.stack(agent_obs, dim=0).to(device)  # Shape: [num_envs * num_agents, obs_size]

    # Number of updates
    num_updates = total_timesteps // (n_steps * num_envs)

    for update in range(num_updates):
        # Reset rollout data
        rollout_data = {
            'obs': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'dones': [],
            'entropies': [],
        }

        for step in range(n_steps):
            # Forward pass: obtain action distributions and agent features
            agent_obs = agent_obs.to(device)
            action_dists, agent_features = actor_model(agent_obs)
            values = critic_model(agent_features)

            # Sample actions
            actions_sample = action_dists.sample()
            action_log_probs = action_dists.log_prob(actions_sample).sum(-1, keepdim=True)
            entropies = action_dists.entropy().sum(-1, keepdim=True)

            # Prepare actions for the environment
            # We need to group actions back to environments
            actions_env = []
            idx = 0
            for _ in range(num_envs):
                env_actions = []
                for _ in range(num_agents):
                    env_actions.append(actions_sample[idx])
                    idx += 1
                actions_env.append(env_actions)

            # Step environment
            obs_next, rewards, dones, infos = env.step(actions_env)
            # obs_next is a list of length num_envs, each element is a list of observations for each agent

            # Flatten next observations
            next_agent_obs = []
            for env_obs in obs_next:
                for obs in env_obs:
                    next_agent_obs.append(obs)
            next_agent_obs = torch.stack(next_agent_obs, dim=0).to(device)

            # Flatten rewards and dones
            rewards_tensor = []
            dones_tensor = []
            for env_rewards in rewards:
                for reward in env_rewards:
                    rewards_tensor.append(reward)
            for env_dones in dones:
                for done in env_dones:
                    dones_tensor.append(done)
            rewards_tensor = torch.tensor(rewards_tensor, dtype=torch.float32, device=device).unsqueeze(-1)
            dones_tensor = torch.tensor(dones_tensor, dtype=torch.float32, device=device).unsqueeze(-1)

            # Store rollout data
            rollout_data['obs'].append(agent_obs)
            rollout_data['actions'].append(actions_sample)
            rollout_data['log_probs'].append(action_log_probs)
            rollout_data['values'].append(values)
            rollout_data['rewards'].append(rewards_tensor)
            rollout_data['dones'].append(dones_tensor)
            rollout_data['entropies'].append(entropies)

            # Update observations for the next step
            agent_obs = next_agent_obs

        # Convert rollout data to tensors
        for key in rollout_data:
            rollout_data[key] = torch.cat(rollout_data[key], dim=0)

        # Compute returns and advantages
        with torch.no_grad():
            next_value = critic_model(actor_model(rollout_data['obs'][-num_envs * num_agents:])[1])
            next_value = next_value.squeeze(-1)
            returns = torch.zeros_like(rollout_data['rewards'])
            advantages = torch.zeros_like(rollout_data['rewards'])
            last_gae_lam = 0
            for t in reversed(range(n_steps * num_envs * num_agents)):
                if t == n_steps * num_envs * num_agents - 1:
                    next_non_terminal = 1.0 - rollout_data['dones'][t]
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - rollout_data['dones'][t + 1]
                    next_values = rollout_data['values'][t + 1]
                delta = rollout_data['rewards'][t] + gamma * next_values * next_non_terminal - rollout_data['values'][t]
                advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            returns = advantages + rollout_data['values']

        # Flatten the batch
        batch_obs = rollout_data['obs']
        batch_actions = rollout_data['actions']
        batch_log_probs = rollout_data['log_probs']
        batch_returns = returns
        batch_advantages = advantages
        batch_values = rollout_data['values']
        batch_entropies = rollout_data['entropies']

        # Normalize advantages
        batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

        # Prepare dataset
        dataset = torch.utils.data.TensorDataset(
            batch_obs,
            batch_actions,
            batch_log_probs,
            batch_returns,
            batch_advantages,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=minibatch_size,
            shuffle=True,
        )

        # PPO policy optimization
        total_value_loss = 0
        total_action_loss = 0
        total_entropy = 0
        num_updates_per_epoch = 0
        for epoch in range(epochs):
            for batch in dataloader:
                batch_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages = [b.to(device) for b in batch]

                # Forward pass
                action_dists, agent_features = actor_model(batch_obs)
                values = critic_model(agent_features).squeeze(-1)

                # Compute action log probabilities and entropy
                action_log_probs = action_dists.log_prob(batch_actions).sum(-1)
                entropy = action_dists.entropy().sum(-1).mean()

                # Compute ratios for PPO
                ratios = torch.exp(action_log_probs - batch_old_log_probs.squeeze(-1))
                surr1 = ratios * batch_advantages.squeeze(-1)
                surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * batch_advantages.squeeze(-1)
                action_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_returns.squeeze(-1))

                # Total loss
                loss = action_loss + value_loss_coef * value_loss - entropy_coef * entropy

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
                total_action_loss += action_loss.item()
                total_entropy += entropy.item()
                num_updates_per_epoch += 1

        # Calculate average losses
        avg_value_loss = total_value_loss / num_updates_per_epoch
        avg_action_loss = total_action_loss / num_updates_per_epoch
        avg_entropy = total_entropy / num_updates_per_epoch

        # Logging metrics to wandb
        total_episode_rewards = rollout_data['rewards'].sum().item() / (num_envs * num_agents)
        wandb.log({
            'update': update,
            'average_reward_per_env': total_episode_rewards,
            'value_loss': avg_value_loss,
            'action_loss': avg_action_loss,
            'entropy': avg_entropy,
            'learning_rate': lr,
        })

        # Print logging information
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Average Reward per Env: {total_episode_rewards:.2f}")

        # Save the model if it achieves a new best average reward
        if total_episode_rewards > best_avg_reward:
            best_avg_reward = total_episode_rewards
            torch.save(
                {'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()},
                'ppo_model_best.pth',
            )
            print(f"New best model saved with average reward: {best_avg_reward:.2f}")

    # Final save of the model
    torch.save(
        {'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()},
        'ppo_model_final.pth',
    )

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    train()
