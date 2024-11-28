# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import wandb
from vmas import make_env
from vmas.simulator.environment import Environment
from model import Actor, Critic  # Ensure you have defined Actor and Critic models in model.py



class CustomEnvironment(Environment):
    def __init__(self, scenario, num_envs, n_agents, device='cpu', **kwargs):
        # Store num_envs and n_agents explicitly
        self.num_envs = num_envs
        self.n_agents = n_agents
        super().__init__(scenario, n_envs=num_envs, device=device, **kwargs)

    def step(self, actions):
        # Call the superclass step method
        obs, rewards, dones, infos = super().step(actions)

        # Handle infos
        if 'infos' in self.scenario.world.data:
            infos = self.scenario.world.data['infos']
        else:
            # Default info structure if none exists
            infos = [[{} for _ in range(self.n_agents)] for _ in range(self.num_envs)]

        return obs, rewards, dones, infos


def make_custom_env(scenario, num_envs, n_agents, device='cpu', **kwargs):
    # Create the base VMAS environment
    env = make_env(scenario=scenario, num_envs=num_envs, device=device, **kwargs)

    # Wrap the environment in the CustomEnvironment class
    custom_env = CustomEnvironment(env.scenario, num_envs=num_envs, n_agents=n_agents, device=device, **kwargs)
    return custom_env

def train(
    env_name='v2',
    num_envs=128,
    n_agents=2,
    n_steps=64,
    total_timesteps=10000000,
    gamma=0.99,
    lam=0.95,
    lr=3e-6,
    clip_param=0.2,
    value_loss_coef=0.9,
    entropy_coef=0.001,
    max_grad_norm=0.05,
    epochs=10,
    minibatch_size=256,
    hidden_size=128,
    device='cuda' if torch.cuda.is_available() else 'cpu',
):
    # Initialize wandb
    wandb.init(
        project="multi-agent-ppo",
        config={
            "env_name": env_name,
            "num_envs": num_envs,
            "n_agents": n_agents,
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

    # Create environment with specified number of agents
    env = make_custom_env(
        scenario=env_name,
        num_envs=num_envs,
        n_agents=n_agents,
        device=device,
        continuous_actions=True,
    )

    # Get observation and action sizes
    obs_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].shape[0]

    # Initialize separate actor and critic models and optimizer
    actor_model = Actor(obs_size, action_size, hidden_size=hidden_size).to(device)
    critic_model = Critic(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(list(actor_model.parameters()) + list(critic_model.parameters()), lr=lr)

    # Watch models with wandb
    wandb.watch(actor_model, log="all")
    wandb.watch(critic_model, log="all")

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
        # Reset reward components
        reward_components = {}

        for step in range(n_steps):
            obs[step].copy_(current_obs)

            # Forward pass: obtain decentralized policy and centralized value
            with torch.no_grad():
                dist = actor_model(current_obs)  # Action distribution, Shape: [num_envs * n_agents, action_size]

                # Get shared features for each agent
                agent_features = actor_model.actor_shared(current_obs)  # Shape: [num_envs * n_agents, hidden_size]

                # Reshape to [num_envs, n_agents, hidden_size] and aggregate
                agent_features = agent_features.view(num_envs, n_agents, hidden_size).mean(dim=1)  # Shape: [num_envs, hidden_size]

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
            obs_next, reward, done, infos = env.step(actions_list)
            reward = torch.cat(reward, dim=0).to(device)  # Shape: [num_envs * n_agents,]
            done = torch.cat(done, dim=0).float().to(device) if isinstance(done, (list, tuple)) else done.float().to(device)  # Shape: [num_envs * n_agents,]

            if done.shape[0] == num_envs:
                done = done.repeat_interleave(n_agents)
            elif done.shape[0] != num_envs * n_agents:
                raise ValueError(f"Unexpected `done` shape: {done.shape}")

            rewards[step].copy_(reward)
            dones[step].copy_(done)

            # Collect per-agent reward components from infos
            for env_idx, env_infos in enumerate(infos):
                for agent_idx, agent_info in enumerate(env_infos):
                    if 'reward_components' in agent_info:
                        components = agent_info['reward_components']
                        for key, value in components.items():
                            if key not in reward_components:
                                reward_components[key] = 0.0
                            # Accumulate the component value
                            reward_components[key] += value

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

        # Initialize variables to accumulate losses and gradient norms
        total_value_loss = 0.0
        total_action_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        total_grad_norms = 0.0
        num_updates_per_epoch = 0

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

                batch_envs = agent_features.size(0) // n_agents  # e.g., 256 /2=128
                agent_features = agent_features.view(batch_envs, n_agents, hidden_size).mean(dim=1)  # Shape: [batch_envs, hidden_size]

                # Value prediction for each environment in the batch
                values_pred = critic_model(agent_features).squeeze(-1)  # Shape: [batch_envs]

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
                # Compute gradient norms
                total_grad_norm = 0.0
                for p in list(actor_model.parameters()) + list(critic_model.parameters()):
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_grad_norm += param_norm.item() ** 2
                total_grad_norm = total_grad_norm ** 0.5

                nn.utils.clip_grad_norm_(list(actor_model.parameters()) + list(critic_model.parameters()), max_grad_norm)
                optimizer.step()

                # Accumulate metrics
                total_value_loss += value_loss.item()
                total_action_loss += action_loss.item()
                total_entropy += entropy.item()
                total_loss += loss.item()
                total_grad_norms += total_grad_norm
                num_updates_per_epoch += 1

        # Calculate average losses and gradient norms
        avg_value_loss = total_value_loss / num_updates_per_epoch
        avg_action_loss = total_action_loss / num_updates_per_epoch
        avg_entropy = total_entropy / num_updates_per_epoch
        avg_loss = total_loss / num_updates_per_epoch
        avg_grad_norm = total_grad_norms / num_updates_per_epoch

        # Compute total reward per agent
        total_reward = rewards.sum().item() / (num_envs * n_agents * n_steps)

        # Compute average reward components
        avg_reward_components = {}
        total_agents = num_envs * n_agents * n_steps
        for key, value in reward_components.items():
            avg_reward_components[key] = value / total_agents

        # Prepare metrics for logging
        metrics = {
            'update': update,
            'average_reward_per_agent': total_reward,
            'value_loss': avg_value_loss,
            'action_loss': avg_action_loss,
            'loss': avg_loss,
            'entropy': avg_entropy,
            'learning_rate': lr,
            'avg_grad_norm': avg_grad_norm,
        }

        # Add average reward components
        for key, value in avg_reward_components.items():
            metrics[f'avg_reward_component/{key}'] = value

        # Log metrics to wandb
        wandb.log(metrics)

        # Logging and saving the best model
        if update % 10 == 0:
            print(f"Update {update}/{num_updates}, Total Reward per Agent: {total_reward:.2f}")

        # Save the model if it achieves a new best reward
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save({'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()}, 'ppo_feedforward_model_best.pth')
            print(f"New best model saved with reward: {best_reward:.2f}")

    # Final save of the model
    torch.save({'actor': actor_model.state_dict(), 'critic': critic_model.state_dict()}, 'ppo_feedforward_model_final.pth')

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    train()
