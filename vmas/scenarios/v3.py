import torch
import math
from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # Default parameters
        n_agents = kwargs.pop("n_agents", 4)
        capacity = kwargs.pop("capacity", 1)
        self.n_packages = kwargs.pop("n_packages", 1)
        self.package_width = kwargs.pop("package_width", 0.15)
        self.package_length = kwargs.pop("package_length", 0.15)
        self.package_mass = kwargs.pop("package_mass", 5)
        self.use_lidar = kwargs.pop("use_lidar", True)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 15)
        self.lidar_range = kwargs.pop("lidar_range", 1.0)
        self.world_semidim = kwargs.pop("world_semidim", 1.5)
        self.agent_radius = kwargs.pop("agent_radius", 0.02)
        self.enable_walls = kwargs.pop("enable_walls", False)
        self.num_walls = kwargs.pop("num_walls", 0)
        self.shaping_factor = 100
        self.energy_coeff = 0.075

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Create world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim,
            y_semidim=self.world_semidim,
            substeps=7,
            drag=0.25,
        )

        # Add agents
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=capacity,
                sensors=[
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.lidar_range,
                        entity_filter=lambda e: e.name.startswith("wall") or e.name.startswith("package"),
                        render_color=Color.GREEN,
                    )
                ] if self.use_lidar else [],
            )
            world.add_agent(agent)

        # Add goal
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)

        # Add packages with attributes
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
            package.on_goal = torch.zeros(batch_dim, dtype=torch.bool, device=device)
            package.prev_dist_to_goal = torch.zeros(batch_dim, device=device)
            self.packages.append(package)
            world.add_landmark(package)

        # Add walls if enabled
        self.walls = []
        if self.enable_walls and self.num_walls > 0:
            for i in range(self.num_walls):
                wall = Landmark(
                    name=f"wall_{i}",
                    collide=True,
                    movable=False,
                    shape=Box(length=0.5, width=0.1),
                    color=Color.BLACK,
                )
                self.walls.append(wall)
                world.add_landmark(wall)

        return world

    def reset_world_at(self, env_index: int = None):
        # Spawn agents
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            min_dist_between_entities=self.agent_radius * 2,
            x_bounds=(-self.world_semidim, -self.world_semidim / 2),
            y_bounds=(-self.world_semidim, self.world_semidim),
            env_index=env_index,
        )

        # Get occupied positions by agents
        agent_occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            agent_occupied_positions = agent_occupied_positions[env_index].unsqueeze(0)

        # Spawn goal
        goal = self.world.landmarks[0]
        ScenarioUtils.spawn_entities_randomly(
            [goal],
            self.world,
            min_dist_between_entities=0.0,
            x_bounds=(self.world_semidim / 2, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_occupied_positions,
            env_index=env_index,
        )

        # Spawn packages and initialize their shaping
        for package in self.packages:
            ScenarioUtils.spawn_entities_randomly(
                [package],
                self.world,
                min_dist_between_entities=0.0,
                x_bounds=(-self.world_semidim / 2, 0),
                y_bounds=(-self.world_semidim, self.world_semidim),
                occupied_positions=agent_occupied_positions,
                env_index=env_index,
            )
            package.on_goal = self.world.is_overlapping(package, package.goal)
            package.dist_to_goal = torch.linalg.vector_norm(
                package.state.pos - package.goal.state.pos, dim=1
            )
            package.prev_dist_to_goal = package.dist_to_goal.clone()

    def observation(self, agent: Agent):
        # Agent's position, velocity, and capacity
        pos = agent.state.pos  # [batch_dim, 2]
        vel = agent.state.vel  # [batch_dim, 2]
        capacity = torch.full((self.world.batch_dim, 1), agent.u_multiplier, device=self.world.device)

        obs = [pos, vel, capacity]

        # Lidar readings
        if self.use_lidar:
            lidar_measurements = agent.sensors[0].measure()  # [batch_dim, n_lidar_rays]
            obs.append(lidar_measurements)

        # Package information
        for package in self.packages:
            pos_to_package = package.state.pos - agent.state.pos
            vel_of_package = package.state.vel
            on_goal = package.on_goal.unsqueeze(-1).float()
            obs.extend([pos_to_package, vel_of_package, on_goal])

        # Agent's relative position to the goal
        agent_to_goal = agent.state.pos - self.world.landmarks[0].state.pos
        obs.append(agent_to_goal)

        # Global goal indicator
        all_packages_on_goal = torch.all(
            torch.stack([package.on_goal for package in self.packages], dim=1), dim=-1
        )
        goal_indicator = all_packages_on_goal.unsqueeze(-1).float()
        obs.append(goal_indicator)

        # Previous action taken by the agent
        if hasattr(agent, 'last_action'):
            obs.append(agent.last_action)
        else:
            # Initialize last_action to zeros
            agent.last_action = torch.zeros(
                (self.world.batch_dim, self.world.dim_p), device=self.world.device
            )
            obs.append(agent.last_action)

        return torch.cat(obs, dim=-1)


    def reward(self, agent: Agent):
        # Only compute reward once per timestep, based on the first agent in the list
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

            # Initialization for rewards
            goal_bonus = 100.0  # Bonus for reaching the goal
            collision_penalty = -1.0  # Penalty for collisions
            proximity_reward_coeff = 10.0  # Reward coefficient for proximity to the package
            shaping_factor = self.shaping_factor  # Shaping factor for package movement
            step_penalty = -1  # Penalty for each timestep
            zone_radius = 0.5  # Zone radius around the package
            outside_zone_penalty = -5.0  # Penalty for agents outside the zone

            # Initialize penalty accumulator
            agent_zone_penalty = torch.zeros(self.world.batch_dim, device=self.world.device)

            for package in self.packages:
                # Calculate distance to goal
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)

                # Color packages based on their goal status
                package.color = torch.tensor(
                    Color.RED.value, device=self.world.device, dtype=torch.float32
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value, device=self.world.device, dtype=torch.float32
                )

                # Shaping reward for package movement
                shaping_reward = (package.prev_dist_to_goal - package.dist_to_goal) * shaping_factor
                self.rew += shaping_reward

                # Update the package's previous distance
                package.prev_dist_to_goal = package.dist_to_goal.clone()

                # Goal bonus for reaching the goal
                self.rew += goal_bonus * package.on_goal.float()

                # Penalty for agents outside the zone
                for agent in self.world.agents:
                    agent_to_package_dist = torch.linalg.vector_norm(
                        agent.state.pos - package.state.pos, dim=1
                    )
                    in_zone = agent_to_package_dist < zone_radius
                    agent_zone_penalty += (~in_zone).float() * outside_zone_penalty

                    # Proximity reward for agents near the package
                    proximity_reward = proximity_reward_coeff * torch.exp(-agent_to_package_dist)
                    self.rew += proximity_reward

            # Apply accumulated zone penalties
            self.rew += agent_zone_penalty

            # Energy penalty for each agent's movement
            energy_rew = self.energy_coeff * -torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(-1)
            self.rew += energy_rew

            # Collision penalty between agents
            collision_count = torch.zeros(self.world.batch_dim, device=self.world.device)
            for i, agent_a in enumerate(self.world.agents):
                for agent_b in self.world.agents[i + 1 :]:
                    is_collision = self.world.is_overlapping(agent_a, agent_b)
                    collision_count += is_collision.float()
            self.rew += collision_penalty * collision_count

            # Step penalty
            self.rew += step_penalty

        return self.rew

    def done(self):
        # Check if all packages are on goal
        all_on_goal = torch.all(
            torch.stack([package.on_goal for package in self.packages], dim=1), dim=-1
        )
        return all_on_goal.to(self.world.device)  # Return termination status

    def set_walls(self, enable: bool, num_walls: int = 0):
        self.enable_walls = enable
        self.num_walls = num_walls

    def pre_step(self, agents):
        # Update last actions for all agents
        for agent in self.world.agents:
            if hasattr(agent, 'action') and agent.action.u is not None:
                agent.last_action = agent.action.u.clone()
            else:
                agent.last_action = torch.zeros(
                    (self.world.batch_dim, self.world.dim_p), device=self.world.device
                )




if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
