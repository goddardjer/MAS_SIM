import torch
from vmas import make_env
from model import SharedActorCritic
import warnings
from vmas.simulator.utils import save_video
import cv2

warnings.filterwarnings("ignore")

def visualize(
    env_name='v2',
    n_agents=4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_render=True,
    filename='evaluation_video.mp4',
    max_steps=500,
    use_lidar=True,
    n_lidar_rays=15,
    n_packages=1,
):
    # Create the environment with the same parameters as training
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
    obs_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].shape[0]
    print(f"Observation Size: {obs_size}, Action Size: {action_size}")

    # Load the trained model
    model = SharedActorCritic(obs_size, action_size, n_agents).to(device)
    model.load_state_dict(torch.load('ppo_feedforward_model_best.pth', map_location=device))
    model.eval()

    # Reset the environment
    current_obs = env.reset()
    current_obs = torch.cat(current_obs, dim=0)

    frame_list = []
    step = 0
    done = False

    while not done and step < max_steps:
        with torch.no_grad():
            # Forward pass through the model for decentralized action
            dist, _ = model(current_obs, centralized=False)
            action = dist.sample()
            
            # Clamp actions to valid range
            action = torch.clamp(action, -1.0, 1.0)

        # Reshape actions for environment
        actions_env = torch.split(action, 1, dim=0)
        actions_list = [actions_env[i] for i in range(n_agents)]

        # Step the environment
        obs_next, reward, done, info = env.step(actions_list)

        # Render the environment and collect frames
        frame = env.render(mode='rgb_array', visualize_when_rgb=False)
        frame_list.append(frame)

        # Display live using OpenCV
        cv2.imshow("Simulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Prepare for the next step
        current_obs = torch.cat(obs_next, dim=0)

        # Check if done is a tensor
        if isinstance(done, (list, tuple)):
            done = torch.cat(done, dim=0).float().to(device)
        else:
            done = done.float().to(device)

        # Terminate if all agents are done
        if done.all():
            break

        step += 1

    # Save the video
    if save_render:
        fps = 5 / env.scenario.world.dt
        save_video(filename, frame_list, fps=fps)
        print(f"Video saved as {filename}")

    # Release OpenCV window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize()
