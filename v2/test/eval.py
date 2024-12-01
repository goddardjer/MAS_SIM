import torch
from torchrl.envs.libs.vmas import VmasEnv
from model import TensorDictModule  # Import only the policy model
import cv2
from vmas.simulator.utils import save_video
import argparse
from matplotlib import pyplot as plt
import os

def evaluate(
    model_path="model.pth",
    env_name="navigation",
    n_agents=3,
    max_steps=500,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_render=True,
    filename="evaluation_video.mp4",
):
    # Load the VMAS environment
    env = VmasEnv(
        scenario=env_name,
        num_envs=1,
        continuous_actions=True,
        max_steps=max_steps,
        device=device,
        n_agents=n_agents,
    )

    print(f"Environment: {env_name}, Number of agents: {n_agents}")
    print("Observation space:", env.observation_spec)
    print("Action space:", env.action_spec)

    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    policy_model = TensorDictModule(
        None,  # Placeholder; will be loaded from state dict
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "action")],
    )
    policy_model.load_state_dict(checkpoint["policy_state_dict"])
    policy_model.to(device)
    policy_model.eval()

    # Reset environment
    current_obs = env.reset()
    frame_list = []  # List to store frames for video
    step = 0
    done = False

    while not done and step < max_steps:
        with torch.no_grad():
            # Get actions from the policy model
            actions = policy_model(current_obs)
            actions = actions[("agents", "action")]

        # Step the environment
        obs_next = env.step(actions)

        # Render and collect frames
        frame = env.render(mode="rgb_array")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_list.append(frame_bgr)

        # Display the frame live using OpenCV
        cv2.imshow("Simulation", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit visualization
            break

        # Update observation
        current_obs = obs_next
        done = current_obs.get(("agents", "done")).all()

        step += 1

    # Save the video
    if save_render:
        fps = 5 / env.scenario.world.dt if hasattr(env.scenario.world, "dt") else 5
        save_video(filename, frame_list, fps=fps)
        print(f"Video saved as {filename}")

    # Close OpenCV window
    cv2.destroyAllWindows()

    print(f"Evaluation completed. Total steps: {step}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained multi-agent PPO model.")
    parser.add_argument("--model_path", type=str, default="model.pth",
                        help="Path to the trained model file.")
    parser.add_argument("--env_name", type=str, default="navigation",
                        help="Scenario name for VMAS.")
    parser.add_argument("--n_agents", type=int, default=3,
                        help="Number of agents.")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum steps per episode.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation.")
    parser.add_argument("--save_render", action="store_true",
                        help="Save the rendered video.")
    parser.add_argument("--filename", type=str, default="evaluation_video.mp4",
                        help="Filename for the saved video.")
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        env_name=args.env_name,
        n_agents=args.n_agents,
        max_steps=args.max_steps,
        device=args.device,
        save_render=args.save_render,
        filename=args.filename,
    )
