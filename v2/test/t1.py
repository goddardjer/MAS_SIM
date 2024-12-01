import argparse
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import multiprocessing
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from matplotlib import pyplot as plt
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-agent PPO agent using TorchRL.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training.")
    parser.add_argument("--frames_per_batch", type=int, default=6000,
                        help="Number of frames collected per training iteration.")
    parser.add_argument("--n_iters", type=int, default=10,
                        help="Number of training iterations.")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of optimization epochs per iteration.")
    parser.add_argument("--minibatch_size", type=int, default=400,
                        help="Size of mini-batches for optimization.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm for clipping.")
    parser.add_argument("--clip_epsilon", type=float, default=0.2,
                        help="Clip value for PPO loss.")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--lmbda", type=float, default=0.9,
                        help="Lambda for GAE.")
    parser.add_argument("--entropy_eps", type=float, default=1e-4,
                        help="Entropy coefficient in PPO loss.")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum steps per episode.")
    parser.add_argument("--scenario_name", type=str, default="navigation",
                        help="Scenario name for VMAS.")
    parser.add_argument("--n_agents", type=int, default=3,
                        help="Number of agents.")
    parser.add_argument("--share_policy", action="store_true",
                        help="Whether to share policy parameters among agents.")
    parser.add_argument("--share_critic", action="store_true",
                        help="Whether to share critic parameters among agents.")
    parser.add_argument("--mappo", action="store_true",
                        help="Use MAPPO (centralized critic). If False, use IPPO (independent critic).")
    parser.add_argument("--save_model", type=str, default="model.pth",
                        help="Path to save the trained model.")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load a pre-trained model.")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment after training.")
    return parser.parse_args()

def build_env(args, device):
    num_vmas_envs = args.frames_per_batch // args.max_steps
    env = VmasEnv(
        scenario=args.scenario_name,
        num_envs=num_vmas_envs,
        continuous_actions=True,
        max_steps=args.max_steps,
        device=device,
        n_agents=args.n_agents,
    )
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    check_env_specs(env)
    return env

def build_policy(env, device, share_parameters_policy):
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    return policy

def build_critic(env, device, share_parameters_critic, mappo):
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )
    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    return critic

def train(args):
    # Set device
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(args.device) if not is_fork else torch.device("cpu")
    vmas_device = device

    # Set seeds for reproducibility
    torch.manual_seed(0)

    # Build environment
    env = build_env(args, vmas_device)

    # Build policy and critic
    policy = build_policy(env, device, args.share_policy)
    critic = build_critic(env, device, args.share_critic, args.mappo)

    # Optionally load model
    if args.load_model and os.path.exists(args.load_model):
        checkpoint = torch.load(args.load_model)
        policy.load_state_dict(checkpoint['policy_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Loaded model from {args.load_model}")

    # Create data collector
    total_frames = args.frames_per_batch * args.n_iters
    collector = SyncDataCollector(
        env,
        policy,
        device=vmas_device,
        storing_device=device,
        frames_per_batch=args.frames_per_batch,
        total_frames=total_frames,
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(args.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch_size,
    )

    # Create loss module
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_eps,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda
    )
    GAE = loss_module.value_estimator

    # Create optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    # Training loop
    pbar = tqdm(total=args.n_iters, desc="Training")
    episode_reward_mean_list = []
    for iteration, tensordict_data in enumerate(collector):
        tensordict_data.set(
            ("next", "agents", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )
        tensordict_data.set(
            ("next", "agents", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
        )

        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(args.num_epochs):
            for _ in range(args.frames_per_batch // args.minibatch_size):
                subdata = replay_buffer.sample()
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), args.max_grad_norm
                )

                optim.step()
                optim.zero_grad()

        collector.update_policy_weights_()

        # Logging
        done = tensordict_data.get(("next", "agents", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
        )
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"Iteration {iteration+1}/{args.n_iters}, Episode Reward Mean: {episode_reward_mean:.2f}")
        pbar.update()

    pbar.close()

    # Save the model
    if args.save_model:
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'critic_state_dict': critic.state_dict(),
        }, args.save_model)
        print(f"Model saved to {args.save_model}")

    # Plot training progress
    plt.plot(episode_reward_mean_list)
    plt.xlabel("Training Iterations")
    plt.ylabel("Episode Reward Mean")
    plt.title("Training Progress")
    plt.show()

    # Optionally render the environment
    if args.render:
        with torch.no_grad():
            env.rollout(
                max_steps=args.max_steps,
                policy=policy,
                callback=lambda env, _: env.render(),
                auto_cast_to_device=True,
                break_when_any_done=False,
            )

def main():
    args = parse_args()
    train(args)

if __name__ == "__main__":
    main()
