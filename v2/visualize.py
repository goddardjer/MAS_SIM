import torch
from vmas import make_env
from model import ActorCriticFeedforward
import warnings
from vmas.simulator.utils import save_video
import cv2  # For displaying frames live

warnings.filterwarnings("ignore")

def visualize(
    env_name='new',
    n_agents=4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_render=True,
    filename='trained_policy.mp4',
    max_steps=500,
    use_lidar=True,
    n_lidar_rays=15,
    n_packages=1,
):
    # Create the environment with the same parameters as during training
    env = make_env(
        scenario=env_name,
        num_envs=1,
        device=device,
        continuous_actions=True,
        n_agents=n_agents,
        use_lidar=use_lidar,
        n_lidar_rays=n_lidar_rays,
        n_packages=n_packages,
    )

    # Dynamically set observation and action sizes
    obs_size = 27  # Manually override to ensure compatibility with the model checkpoint
    action_size = env.action_space[0].shape[0]

    print(f"Manually Set Observation Size: {obs_size}, Action Size: {action_size}")

    # Load the trained feedforward model
    model = ActorCriticFeedforward(obs_size, action_size).to(device)
    model.load_state_dict(torch.load('ppo_feedforward_model_best.pth', map_location=device))
    model.eval()

    # Reset the environment
    current_obs = env.reset()
    current_obs = torch.cat(current_obs, dim=0)

    # Add padding if current_obs is of size 26 to match expected size of 27
    if current_obs.shape[-1] == 26:
        current_obs = torch.cat([current_obs, torch.zeros(current_obs.shape[0], 1, device=device)], dim=-1)

    frame_list = []  # List to store frames for video
    step = 0
    done = False

    while not done and step < max_steps:
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

        # Render the environment and collect frames
        frame = env.render(mode='rgb_array', visualize_when_rgb=False)
        frame_list.append(frame)

        # Display the frame live using OpenCV
        cv2.imshow("Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit visualization
            break

        # Prepare for the next step
        current_obs = torch.cat(obs_next, dim=0)

        # Add padding if current_obs is of size 26 to match expected size of 27
        if current_obs.shape[-1] == 26:
            current_obs = torch.cat([current_obs, torch.zeros(current_obs.shape[0], 1, device=device)], dim=-1)

        # Check if done is already a tensor
        if isinstance(done, (list, tuple)):
            done = torch.cat(done, dim=0).float().to(device)
        else:
            done = done.float().to(device)

        # Break loop if all agents are done
        if done.all():
            break

        step += 1

    # Save the video
    if save_render:
        fps = 30 / env.scenario.world.dt
        save_video(filename, frame_list, fps=fps)
        print(f"Video saved as {filename}")

    # Release OpenCV window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize()
