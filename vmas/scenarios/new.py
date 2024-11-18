import torch
import math
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.joints import Joint
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Extract parameters
        n_agents = kwargs.pop("n_agents", 2)
        capacity = kwargs.pop("capacity", 1.0)
        self.n_packages = kwargs.pop("n_packages", 1)
        self.package_width = kwargs.pop("package_width", 0.15)
        self.package_length = kwargs.pop("package_length", 0.15)
        self.package_mass = kwargs.pop("package_mass", 20)
        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.shaping_factor = 100
        self.world_semidim = 1
        self.agent_radius = 0.03

        self.energy_coeff = 0.075
        self.energy_rew = torch.zeros(batch_dim, device=device)

        # Create the world
        world = World(
            batch_dim,
            device,
            x_semidim=2 * self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
            y_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
            substeps=7,
            drag=0.25,
        )
        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=capacity,
            )
            world.add_agent(agent)

        # Add goal landmark
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)

        # Add packages
        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package_{i}",
                collide=True,
                movable=True,
                mass=self.package_mass,
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
            self.packages.append(package)
            world.add_landmark(package)

        # Initialize maximum episode length and step counter
        self.max_episode_length = 500  # Adjust as needed
        self.current_step = torch.zeros(batch_dim, device=device, dtype=torch.int32)

        return world

    def reset_world_at(self, env_index: int = None):
        # Reset step counter
        if env_index is None:
            self.current_step = torch.zeros_like(self.current_step)
        else:
            self.current_step[env_index] = 0

        # Randomly place agents
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=self.agent_radius * 5,
            x_bounds=(-1.75, -1.0),
            y_bounds=(-self.world_semidim, self.world_semidim),
        )

        agent_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            agent_positions = agent_positions[env_index].unsqueeze(0)

        goal = self.world.landmarks[0]

        # Randomly place the goal
        ScenarioUtils.spawn_entities_randomly(
            [goal],
            self.world,
            env_index,
            min_dist_between_entities=0.0,
            x_bounds=(1.0, 1.75),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_positions,
        )

        # Randomly place the packages
        ScenarioUtils.spawn_entities_randomly(
            self.packages,
            self.world,
            env_index,
            min_dist_between_entities=0.0,
            x_bounds=(-0.5, 0.5),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_positions,
        )

        # Initialize package shaping and on_goal flags
        for package in self.packages:
            package.on_goal = self.world.is_overlapping(package, package.goal)
            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

        # Initialize agents' package shaping for moving towards the package
        for agent in self.world.agents:
            agent_to_package_dists = torch.stack(
                [
                    torch.linalg.vector_norm(agent.state.pos - p.state.pos, dim=1)
                    for p in self.packages
                ]
            )  # Shape: (num_packages, batch_dim)
            min_distance, _ = agent_to_package_dists.min(dim=0)  # Shape: (batch_dim,)
            agent.package_shaping = min_distance * self.shaping_factor

    def pre_step(self, agents):
        threshold_distance = 0.25
        velocity_threshold = 0.01

        batch_size = self.world.batch_dim

        for batch_idx in range(batch_size):
            for agent in agents:
                agent_speed = torch.linalg.norm(agent.state.vel[batch_idx], dim=-1)
                for package in self.packages:
                    distance = torch.linalg.norm(
                        agent.state.pos[batch_idx] - package.state.pos[batch_idx], dim=-1
                    )

                    # Check interaction conditions for this batch index
                    if distance < threshold_distance and agent_speed < velocity_threshold:
                        # Check if a joint already exists between agent and package in this batch
                        joint_exists = any(
                            joint.entity_a == agent
                            and joint.entity_b == package
                            and joint.batch_index == batch_idx
                            for joint in self.world.joints
                        )
                        if not joint_exists:
                            self.create_joint(
                                agent, package, distance.item(), batch_idx
                            )
                            print(
                                f"Joint created between {agent.name} and {package.name} in batch {batch_idx}"
                            )

    def create_joint(self, agent: Agent, package: Landmark, distance, batch_idx):
        # Ensure anchors are set as constant tuples of floats, not tensors
        anchor_a = (0.0, 0.0)
        anchor_b = (0.0, 0.0)

        # Create a joint for the specific batch index
        joint = Joint(
            entity_a=agent,
            entity_b=package,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            rotate_a=False,
            rotate_b=False,
            dist=max(distance, 0.1),  # Ensure distance is at least 0.1
            collidable=True,
            width=0.05,
            mass=0.5,
        )
        # Assign the batch index to the joint for identification
        joint.batch_index = batch_idx
        self.world.add_joint(joint)
        print(
            f"Joint created between {agent.name} and {package.name} in batch {batch_idx}"
        )

    def reward(self, agent: Agent):
        # Initialize agent reward
        agent_reward = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.float32,
        )

        batch_size = self.world.batch_dim

        # Compute distance to nearest package
        agent_to_package_dists = torch.stack(
            [
                torch.linalg.vector_norm(agent.state.pos - p.state.pos, dim=1)
                for p in self.packages
            ]
        )  # Shape: (num_packages, batch_dim)
        min_distance, _ = agent_to_package_dists.min(dim=0)  # Shape: (batch_dim,)

        # Initialize agent.package_shaping if not present
        if not hasattr(agent, "package_shaping"):
            agent.package_shaping = min_distance * self.shaping_factor

        # Compute shaping difference for moving towards the package
        current_shaping = min_distance * self.shaping_factor
        shaping_diff = agent.package_shaping - current_shaping
        agent_reward += shaping_diff
        agent.package_shaping = current_shaping

        # Process each batch individually
        for batch_idx in range(batch_size):
            # Check if agent is interacting with a package in this batch
            interacting = any(
                joint.entity_a == agent
                and joint.entity_b in self.packages
                and joint.batch_index == batch_idx
                for joint in self.world.joints
            )

            if interacting:
                # Agent is interacting with a package
                for package in self.packages:
                    # Check if agent is attached to this package in this batch
                    if any(
                        joint.entity_a == agent
                        and joint.entity_b == package
                        and joint.batch_index == batch_idx
                        for joint in self.world.joints
                    ):
                        # Compute reward based on package movement
                        if not hasattr(package, "global_shaping"):
                            package.global_shaping = (
                                torch.linalg.vector_norm(
                                    package.state.pos - package.goal.state.pos, dim=1
                                )
                                * self.shaping_factor
                            )

                        prev_package_shaping = package.global_shaping[batch_idx]

                        # Calculate distance to goal
                        package.dist_to_goal = torch.linalg.vector_norm(
                            package.state.pos - package.goal.state.pos, dim=1
                        )
                        # Check if package reached goal
                        package.on_goal[batch_idx] = self.world.is_overlapping(
                            package, package.goal, batch_idx
                        )

                        # Reward shaping: reward progress towards the goal
                        package_shaping = (
                            package.dist_to_goal[batch_idx] * self.shaping_factor
                        )
                        shaping_diff = prev_package_shaping - package_shaping
                        agent_reward[batch_idx] += shaping_diff
                        package.global_shaping[batch_idx] = package_shaping
            else:
                # Penalize agent for not interacting
                agent_reward[batch_idx] -= 0.1  # Penalty value for idleness

        # Energy penalty for the agent
        energy_penalty = (
            self.energy_coeff
            * -torch.linalg.vector_norm(agent.action.u, dim=-1)
            / math.sqrt(self.world.dim_p * ((agent.u_range * agent.u_multiplier) ** 2))
        )
        agent_reward += energy_penalty

        return agent_reward

    def observation(self, agent: Agent):
        batch_size = self.world.batch_dim

        # Initialize interaction flag tensor
        interaction_flag = torch.zeros(
            (batch_size, 1), device=self.world.device, dtype=torch.float32
        )

        # Process each batch individually to check interaction
        for batch_idx in range(batch_size):
            # Check if agent is attached to a package in this batch
            interacting = any(
                joint.entity_a == agent
                and joint.entity_b in self.packages
                and joint.batch_index == batch_idx
                for joint in self.world.joints
            )
            interaction_flag[batch_idx] = float(interacting)

        package_obs = []
        for package in self.packages:
            # Direction from agent to package's goal
            dir_to_goal = package.goal.state.pos - agent.state.pos
            dir_to_goal_norm = dir_to_goal / (
                torch.linalg.norm(dir_to_goal, dim=-1, keepdim=True) + 1e-6
            )

            # Direction from agent to package
            dir_to_package = package.state.pos - agent.state.pos
            dir_to_package_norm = dir_to_package / (
                torch.linalg.norm(dir_to_package, dim=-1, keepdim=True) + 1e-6
            )

            package_obs.extend(
                [
                    dir_to_goal_norm,  # Normalized direction to package goal
                    dir_to_package_norm,  # Normalized direction to package
                    package.state.vel,  # Package velocity
                    package.on_goal.unsqueeze(-1).float(),  # Package on goal
                ]
            )

        # Include observations about other agents
        other_agents_obs = []
        for other_agent in self.world.agents:
            if other_agent != agent:
                # Direction from agent to other agent
                dir_to_other_agent = other_agent.state.pos - agent.state.pos
                dir_to_other_agent_norm = dir_to_other_agent / (
                    torch.linalg.norm(dir_to_other_agent, dim=-1, keepdim=True) + 1e-6
                )

                other_agents_obs.extend(
                    [
                        dir_to_other_agent_norm,  # Normalized direction to other agent
                        other_agent.state.vel,  # Other agent's velocity
                    ]
                )

        # Include agent's own heading
        agent_heading = agent.state.vel / (
            torch.linalg.norm(agent.state.vel, dim=-1, keepdim=True) + 1e-6
        )

        return torch.cat(
            [
                agent.state.pos,  # Agent position
                agent.state.vel,  # Agent velocity
                agent_heading,  # Agent's own heading
                interaction_flag,  # Interaction flag
                *package_obs,  # Package observations
                *other_agents_obs,  # Other agents' observations
            ],
            dim=-1,
        )

    def done(self):
        packages_on_goal = torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )
        max_steps_exceeded = self.current_step >= self.max_episode_length
        return packages_on_goal | max_steps_exceeded
