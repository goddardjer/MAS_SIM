import torch
import math
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.joints import Joint
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Set batch_dim to 1 since we're running a single environment
        batch_dim = 1

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

        # Create the world
        world = World(
            batch_dim=batch_dim,
            device=device,
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
        self.current_step = torch.zeros(batch_dim, dtype=torch.int32, device=device)

        return world

    def reset_world(self):
        # Reset step counter
        self.current_step = torch.zeros_like(self.current_step)

        env_index = 0  # Since batch size is 1

        # Randomly place agents
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index=env_index,
            min_dist_between_entities=self.agent_radius * 5,
            x_bounds=(-1.75, -1.0),
            y_bounds=(-self.world_semidim, self.world_semidim),
        )

        agent_positions = torch.stack(
            [agent.state.pos[env_index] for agent in self.world.agents], dim=0
        )

        goal = self.world.landmarks[0]

        # Randomly place the goal
        ScenarioUtils.spawn_entities_randomly(
            [goal],
            self.world,
            env_index=env_index,
            min_dist_between_entities=0.0,
            x_bounds=(1.0, 1.75),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_positions.unsqueeze(0),
        )

        # Randomly place the packages
        ScenarioUtils.spawn_entities_randomly(
            self.packages,
            self.world,
            env_index=env_index,
            min_dist_between_entities=0.0,
            x_bounds=(-0.5, 0.5),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_positions.unsqueeze(0),
        )

        # Initialize package shaping and on_goal flags
        for package in self.packages:
            package.on_goal = self.world.is_overlapping(
                package, package.goal, env_index=env_index
            )
            package.global_shaping = (
                torch.linalg.vector_norm(
                    package.state.pos[env_index] - package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

        # Initialize agents' package shaping for moving towards the package
        for agent in self.world.agents:
            agent_to_package_dists = torch.stack(
                [
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - p.state.pos[env_index]
                    )
                    for p in self.packages
                ]
            )  # Shape: (num_packages,)
            min_distance, _ = agent_to_package_dists.min(dim=0)
            agent.package_shaping = min_distance * self.shaping_factor

    def reset_world_at(self, env_index: int = None):
        # Since we're running a single environment, we can call reset_world
        self.reset_world()

    def pre_step(self, agents):
        """
        Detect proximity and apply forces to simulate pushing behavior.
        """
        threshold_distance = 0.25
        env_index = 0  # Since batch size is 1

        for agent in agents:
            for package in self.packages:
                distance = torch.linalg.norm(
                    agent.state.pos[env_index] - package.state.pos[env_index]
                )
                
                # Check if agent is within pushing range of the package
                if distance < threshold_distance:
                    # Calculate the direction vector from agent to package
                    direction = (package.state.pos[env_index] - agent.state.pos[env_index])
                    direction_norm = direction / (torch.linalg.norm(direction) + 1e-6)
                    
                    # Apply force based on agent's pushing power and direction
                    push_strength = 0.5  # Adjust pushing force strength as needed
                    package_force = direction_norm * push_strength
                    
                    # Apply the calculated force to the package's velocity
                    package.state.vel[env_index] += package_force
                    # print(f"{agent.name} is pushing {package.name} with force {package_force}")


    def create_joint(self, agent: Agent, package: Landmark, distance):
        # Ensure anchors are set as constant tuples of floats, not tensors
        anchor_a = (0.0, 0.0)
        anchor_b = (0.0, 0.0)

        # Create a joint
        joint = Joint(
            entity_a=agent,
            entity_b=package,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            rotate_a=False,
            rotate_b=False,
            dist=distance,
            collidable=True,
            width=0.05,
            mass=0.5,
        )

        # Set any additional properties needed for full initialization
        if not hasattr(joint, 'rotation'):
            joint.rotation = torch.zeros(2)  # Example placeholder; adjust as necessary

        self.world.add_joint(joint)
        print(f"Joint created between {agent.name} and {package.name}")


    def reward(self, agent: Agent):
        agent_reward = torch.zeros(1, device=self.world.device)
        env_index = 0  # Since batch size is 1

        # Compute distance to nearest package
        agent_to_package_dists = torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos[env_index] - p.state.pos[env_index]
                )
                for p in self.packages
            ]
        )
        min_distance, _ = agent_to_package_dists.min(dim=0)

        # Compute shaping reward for moving towards the package
        current_shaping = min_distance * self.shaping_factor
        if not hasattr(agent, "package_shaping"):
            agent.package_shaping = current_shaping
        shaping_diff = agent.package_shaping - current_shaping
        agent_reward += shaping_diff
        agent.package_shaping = current_shaping

        # Reward for pushing package towards the goal
        for package in self.packages:
            if not hasattr(package, "global_shaping"):
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )

            # Calculate reward based on the package's progress towards the goal
            prev_package_shaping = package.global_shaping
            package.dist_to_goal = torch.linalg.vector_norm(
                package.state.pos[env_index] - package.goal.state.pos[env_index]
            )
            package.on_goal = self.world.is_overlapping(
                package, package.goal, env_index=env_index
            )

            package_shaping = package.dist_to_goal * self.shaping_factor
            shaping_diff = prev_package_shaping - package_shaping
            agent_reward += shaping_diff
            package.global_shaping = package_shaping

        # Energy penalty for exerting force
        energy_penalty = (
            self.energy_coeff
            * -torch.linalg.norm(agent.action.u[env_index])
            / math.sqrt(
                self.world.dim_p * ((agent.u_range * agent.u_multiplier) ** 2)
            )
        )
        agent_reward += energy_penalty

        return agent_reward.squeeze()


    def observation(self, agent: Agent):
        env_index = 0  # Since batch size is 1

        # Initialize interaction flag
        interaction_flag = 0.0

        # Check if agent is attached to a package
        interacting = any(
            joint.entity_a == agent and joint.entity_b in self.packages
            for joint in self.world.joints
        )
        if interacting:
            interaction_flag = 1.0

        package_obs = []
        for package in self.packages:
            # Direction from agent to package's goal
            dir_to_goal = (
                package.goal.state.pos[env_index] - agent.state.pos[env_index]
            )
            dir_to_goal_norm = dir_to_goal / (
                torch.linalg.norm(dir_to_goal) + 1e-6
            )

            # Direction from agent to package
            dir_to_package = (
                package.state.pos[env_index] - agent.state.pos[env_index]
            )
            dir_to_package_norm = dir_to_package / (
                torch.linalg.norm(dir_to_package) + 1e-6
            )

            package_obs.extend(
                [
                    dir_to_goal_norm,  # Normalized direction to package goal
                    dir_to_package_norm,  # Normalized direction to package
                    package.state.vel[env_index],  # Package velocity
                    torch.tensor([float(package.on_goal)], device=self.world.device),  # Package on goal
                ]
            )

        # Include observations about other agents
        other_agents_obs = []
        for other_agent in self.world.agents:
            if other_agent != agent:
                # Direction from agent to other agent
                dir_to_other_agent = (
                    other_agent.state.pos[env_index] - agent.state.pos[env_index]
                )
                dir_to_other_agent_norm = dir_to_other_agent / (
                    torch.linalg.norm(dir_to_other_agent) + 1e-6
                )

                other_agents_obs.extend(
                    [
                        dir_to_other_agent_norm,  # Normalized direction to other agent
                        other_agent.state.vel[env_index],  # Other agent's velocity
                    ]
                )

        # Include agent's own heading
        agent_heading = agent.state.vel[env_index] / (
            torch.linalg.norm(agent.state.vel[env_index]) + 1e-6
        )

        observation = torch.cat(
            [
                agent.state.pos[env_index],  # Agent position
                agent.state.vel[env_index],  # Agent velocity
                agent_heading,  # Agent's own heading
                torch.tensor([interaction_flag], device=self.world.device),  # Interaction flag
                *package_obs,  # Package observations
                *other_agents_obs,  # Other agents' observations
            ],
            dim=-1,
        )
        return observation

    def done(self):
        env_index = 0  # Since batch size is 1
        packages_on_goal = all(package.on_goal for package in self.packages)
        max_steps_exceeded = self.current_step[env_index] >= self.max_episode_length
        return packages_on_goal or max_steps_exceeded
