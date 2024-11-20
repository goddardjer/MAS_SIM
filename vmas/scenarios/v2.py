# v2.py

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
        capacity = kwargs.pop("capacity", 0.8)
        self.n_packages = kwargs.pop("n_packages", 1)
        self.package_width = kwargs.pop("package_width", 0.15)
        self.package_length = kwargs.pop("package_length", 0.15)
        self.package_mass = kwargs.pop("package_mass", 20)
        self.use_lidar = kwargs.pop("use_lidar", True)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 15)
        self.lidar_range = kwargs.pop("lidar_range", 1.0)
        self.world_semidim = kwargs.pop("world_semidim", 1.5)
        self.agent_radius = kwargs.pop("agent_radius", 0.03)
        self.enable_walls = kwargs.pop("enable_walls", False)
        self.num_walls = kwargs.pop("num_walls", 0)
        self.shaping_factor = 100
        self.energy_coeff = 0.075
        self.energy_rew = torch.zeros(batch_dim, device=device)

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

        # Add packages with an `on_goal` attribute and `global_shaping` initialization
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
            package.on_goal = torch.zeros(batch_dim, dtype=torch.bool, device=device)  # Initialize on_goal attribute
            package.global_shaping = torch.zeros(batch_dim, device=device)  # Initialize global_shaping attribute
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
            env_index,
            min_dist_between_entities=self.agent_radius * 2,
            x_bounds=(-self.world_semidim, -self.world_semidim / 2),
            y_bounds=(-self.world_semidim, self.world_semidim),
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
            env_index,
            min_dist_between_entities=0.0,
            x_bounds=(self.world_semidim / 2, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_occupied_positions,
        )

        # Spawn packages and initialize their shaping
        for package in self.packages:
            ScenarioUtils.spawn_entities_randomly(
                [package],
                self.world,
                env_index,
                min_dist_between_entities=0.0,
                x_bounds=(-self.world_semidim / 2, 0),
                y_bounds=(-self.world_semidim, self.world_semidim),
                occupied_positions=agent_occupied_positions,
            )
            package.on_goal = self.world.is_overlapping(package, package.goal)
            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    ) * self.shaping_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos, dim=1
                    ) * self.shaping_factor
                )

    def observation(self, agent: Agent):
        # Agent's position, velocity, and capacity
        pos = agent.state.pos  # [batch_dim, 2]
        vel = agent.state.vel  # [batch_dim, 2]
        capacity = torch.full((self.world.batch_dim, 1), agent.u_multiplier, device=self.world.device)

        obs = [pos, vel, capacity]  # Initial observations: 5 features

        # Lidar readings
        if self.use_lidar:
            lidar_measurements = agent.sensors[0].measure()  # [batch_dim, n_lidar_rays]
            obs.append(lidar_measurements)  # Add lidar readings to obs

        # Package information
        for package in self.packages:
            pos_to_goal = package.state.pos - package.goal.state.pos  # Vector from package to goal
            pos_to_agent = package.state.pos - agent.state.pos        # Vector from package to agent
            package_vel = package.state.vel                           # Package velocity
            on_goal = package.on_goal.unsqueeze(-1)                   # Goal status (boolean as a feature)

            obs.extend([pos_to_goal, pos_to_agent, package_vel, on_goal])  # 8 features per package

        return torch.cat(obs, dim=-1)  # Final concatenated observation tensor

    def reward(self, agent: Agent):
        # Only compute reward once per timestep, based on the first agent in list
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

            # Reward for each package
            for package in self.packages:
                # Calculate distance to goal
                package.dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(
                    Color.RED.value, device=self.world.device, dtype=torch.float32
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value, device=self.world.device, dtype=torch.float32
                )

                # Shaping and reward based on reaching goal
                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal] - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

            # Energy penalty for each agent's movement
            self.energy_rew = self.energy_coeff * -torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(-1)
            self.rew += self.energy_rew

        return self.rew  # Return reward tensor

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
        # Pre-step logic if needed
        pass


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
