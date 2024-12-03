import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

device = torch.device("cpu")
vmas_device = device

max_steps = 1500
scenario_name = "v2"
n_agents = 5
num_vmas_envs = 1

# Observation settings
use_goal_obs = False
use_package_obs = False
use_other_agent_obs = False
use_lidar = True

# Reward settings
use_package_shaping = True
use_agent_shaping = True
use_contribution = False

env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=True,
    max_steps=max_steps,
    device=vmas_device,
    n_agents=n_agents,
    use_goal_obs = use_goal_obs,
    use_package_obs = use_package_obs,
    use_other_agent_obs = use_other_agent_obs,
    use_lidar = use_lidar,
    use_package_shaping = use_package_shaping,
    use_agent_shaping = use_agent_shaping,
    use_contribution = use_contribution,
)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)

share_parameters_policy = True

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

policy.load_state_dict(torch.load(f"../logs/obs_exps/goal-{use_goal_obs}_pkg-{use_package_obs}_agt-{use_other_agent_obs}_lidar-{use_lidar}_rew-Ours.pt", map_location=device))
# policy.load_state_dict(torch.load(f"../logs/rew_exps/G-True_Gs-{use_package_shaping}_As-{use_agent_shaping}_C-{use_contribution}_Obs-Ours.pt", map_location=device))

with torch.no_grad():
    env.rollout(
        max_steps=max_steps,
        policy=policy,
        callback=lambda env, _: env.render(),
        auto_cast_to_device=True,
        break_when_any_done=False,
    )