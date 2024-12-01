import torch
from vmas import make_env
from model import Actor  # Import only the actor
import warnings
from vmas.simulator.utils import save_video
import cv2

warnings.filterwarnings("ignore")

def visualize(
    env_name='v2',
    n_agents=3,  # Set to desired number of agents for evaluation
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_render=True,
    filename='evaluation_video',
    max_steps=500,
    use_lidar=True,
    n_lidar_rays=15,
    n_packages=1,
):
    # Create the environment with the desired number of agents
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

    # Dynamically get observation and action sizes from the environment
    obs_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].shape[0]
    print(f"Observation Size: {obs_size}, Action Size: {action_size}")

    # Initialize the actor model (no need for critic during evaluation)
    actor_model = Actor(obs_size, action_size, hidden_size=128).to(device)

    # Load the trained model's actor state dict
    checkpoint = torch.load('ppo_feedforward_model_best.pth', map_location=device)
    actor_model.load_state_dict(checkpoint['actor'])
    actor_model.eval()

    # Reset the environment
    current_obs = env.reset()
    current_obs = torch.cat(current_obs, dim=0).to(device)  # Shape: [n_agents, obs_size]

    frame_list = []  # List to store frames for video
    step = 0
    done = False

    while not done and step < max_steps:
        with torch.no_grad():
            # Forward pass through the actor to get action distribution
            dist = actor_model(current_obs)
            action = dist.sample()
            
            # Clamp actions to the valid range [-1.0, 1.0]
            action = torch.clamp(action, -1.0, 1.0)

        # Prepare actions for environment
        actions_list = [action[i].unsqueeze(0) for i in range(n_agents)]

        # Step environment
        obs_next, reward, done, info = env.step(actions_list)

        # Render the environment and collect frames
        frame = env.render(mode='rgb_array', visualize_when_rgb=False)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_list.append(frame_bgr)

        # Display the frame live using OpenCV
        cv2.imshow("Simulation", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit visualization
            break

        # Prepare for the next step
        current_obs = torch.cat(obs_next, dim=0).to(device)

        # Check if done is already a tensor and adjust if needed
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
        # Handle cases where env.scenario.world.dt might not exist
        fps = 5 / env.scenario.world.dt if hasattr(env.scenario.world, 'dt') else 5
        save_video(filename, frame_list, fps=fps)
        print(f"Video saved as {filename}")

    # Release OpenCV window
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize()
