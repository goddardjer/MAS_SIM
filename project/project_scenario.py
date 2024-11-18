# project_scenario.py

import torch
import math
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
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

        # Add package
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

        return world

    def reset_world_at(self, env_index: int = None):
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

        # Randomly place the package
        ScenarioUtils.spawn_entities_randomly(
            self.packages,
            self.world,
            env_index,
            min_dist_between_entities=0.0,
            x_bounds=(-0.5, 0.5),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_positions,
        )

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

                package_shaping = package.dist_to_goal * self.shaping_factor
                self.rew += (
                    package.global_shaping - package_shaping
                )
                package.global_shaping = package_shaping

            # Energy penalty
            self.energy_rew = self.energy_coeff * -torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(-1)

            # Total reward
            self.rew += self.energy_rew

        return self.rew

    def observation(self, agent: Agent):
        package_obs = []
        for package in self.packages:
            package_obs.append(package.state.pos - package.goal.state.pos)  # Package to goal
            package_obs.append(package.state.pos - agent.state.pos)         # Package to agent
            package_obs.append(package.state.vel)                           # Package velocity
            package_obs.append(package.on_goal.unsqueeze(-1).float())       # Package on goal

        return torch.cat(
            [
                agent.state.pos,     # Agent position
                agent.state.vel,     # Agent velocity
                *package_obs,
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
    
# if __name__ == "__main__":
#     render_interactively(__file__, control_two_agents=True)
