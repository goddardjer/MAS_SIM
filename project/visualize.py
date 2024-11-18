

import torch
from vmas import make_env
from model import ActorCriticFeedforward
import warnings
warnings.filterwarnings("ignore")


def visualize(env_name='project_scenario', n_agents=5, device='cpu'):
    env = make_env(
        scenario=env_name,
        num_envs=1,
        device=device,
        continuous_actions=True,
        n_agents=n_agents,
    )

    obs_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].shape[0]

    # Load the trained feedforward model
    model = ActorCriticFeedforward(obs_size, action_size).to(device)
    model.load_state_dict(torch.load('ppo_feedforward_model_best.pth', map_location=device))
    # model.load_state_dict(torch.load('ppo_feedforward_model_final.pth', map_location=device))
    model.eval()

    # Reset the environment
    current_obs = env.reset()
    current_obs = torch.cat(current_obs, dim=0)

    done = False
    while not done:
        with torch.no_grad():
            # Forward pass through the model
            dist, _ = model(current_obs)
            action = dist.sample()
            
            # Clamp actions to the valid range [-1.0, 1.0]
            action = torch.clamp(action, -1.0, 1.0)

        # Reshape actions for environment
        actions_env = torch.split(action, 1, dim=0)
        actions_list = [actions_env[i] for i in range(n_agents)]

        # Step environment
        obs_next, reward, done, info = env.step(actions_list)

        # Render the environment
        env.render()

        # Prepare for the next step
        current_obs = torch.cat(obs_next, dim=0)

        # Check if done is already a tensor
        if isinstance(done, (list, tuple)):
            done = torch.cat(done, dim=0).float().to(device)
        else:
            done = done.float().to(device)

        # Break loop if all agents are done
        if done.all():
            break


if __name__ == '__main__':
    visualize()

