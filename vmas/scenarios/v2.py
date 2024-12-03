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
        self.package_mass = kwargs.pop("package_mass", 3)
        self.use_lidar = kwargs.pop("use_lidar", True)
        self.n_lidar_rays = kwargs.pop("n_lidar_rays", 72)
        self.lidar_range = kwargs.pop("lidar_range", 2.0)
        self.world_semidim = kwargs.pop("world_semidim", 1.5)
        self.agent_radius = kwargs.pop("agent_radius", 0.02)
        self.enable_walls = kwargs.pop("enable_walls", False)
        self.num_walls = kwargs.pop("num_walls", 0)
        self.use_package_obs = kwargs.pop("use_package_obs", False)
        self.use_other_agent_obs = kwargs.pop("use_other_agent_obs", False)
        self.use_goal_obs = kwargs.pop("use_goal_obs", True)
        self.shaping_factor = 50
        self.agent_shaping_factor = 20
        self.goal_bonus = 100.0
        self.energy_coeff = 0.1
        self.energy_rew = torch.zeros(batch_dim, device=device)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        # Create world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim,
            y_semidim=self.world_semidim,
            # substeps=7,
            # drag=0.25,
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
                name=f"package {i}",
                collide=True,
                movable=True,
                mass=self.package_mass,
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
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
            min_dist_between_entities=max(
                package.shape.circumscribed_radius() + goal.shape.radius + 0.01
                for package in self.packages
            ),
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

        for agent in self.world.agents:
            if env_index is None:
                agent.global_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - self.packages[0].state.pos, dim=1
                ) * self.agent_shaping_factor
                )
            else:
                agent.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - self.packages[0].state.pos[env_index]
                    ) * self.agent_shaping_factor
                )

    def reward(self, agent: Agent):

        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )

            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(
                    Color.RED.value,
                    device=self.world.device,
                    dtype=torch.float32,
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value,
                    device=self.world.device,
                    dtype=torch.float32,
                )
                for a in self.world.agents:
                    a.dist_to_package = torch.linalg.vector_norm(
                        a.state.pos - package.state.pos, dim=1
                    )

                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

                self.energy_rew = self.energy_coeff * -torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
                    for a in self.world.agents
                ],
                dim=1,
                ).sum(-1)

                self.rew += self.goal_bonus * package.on_goal.float()
                self.rew += self.energy_rew

        agent_shaping = agent.dist_to_package * self.agent_shaping_factor
        rew = agent.global_shaping - agent_shaping
        agent.global_shaping = agent_shaping

        return self.rew + rew

    def observation(self, agent: Agent):
        # Lidar readings
        if self.use_lidar:
            lidar_measurements = agent.sensors[0]._max_range - agent.sensors[0].measure()  # [batch_dim, n_lidar_rays]

        if self.use_package_obs:
            package_obs = []
            for package in self.packages:
                package_obs.append(package.state.pos - package.goal.state.pos)
                package_obs.append(package.state.pos - agent.state.pos)
                package_obs.append(package.state.vel)
                package_obs.append(package.on_goal.unsqueeze(-1))
            package_obs = torch.cat(package_obs, dim=-1)

        if self.use_other_agent_obs:
            other_agent_obs = []
            for other_agent in self.world.agents:
                if other_agent == agent:
                    continue
                other_agent_obs.append(other_agent.state.pos)
                other_agent_obs.append(other_agent.state.vel)
            other_agent_obs = torch.cat(other_agent_obs, dim=-1)

        if self.use_goal_obs:
            goal_obs = []
            for package in self.packages:
                goal_obs.append(package.goal.state.pos)
            goal_obs = torch.cat(goal_obs, dim=-1)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                goal_obs if self.use_goal_obs else torch.tensor([], device=self.world.device),
                package_obs if self.use_package_obs else torch.tensor([], device=self.world.device),
                other_agent_obs if self.use_other_agent_obs else torch.tensor([], device=self.world.device),
                lidar_measurements if self.use_lidar else torch.tensor([], device=self.world.device),
            ],
            dim=-1,
        )

    def done(self):
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )

    def set_walls(self, enable: bool, num_walls: int = 0):
        self.enable_walls = enable
        self.num_walls = num_walls

    def pre_step(self, agents):
        # Pre-step logic if needed
        pass

if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
