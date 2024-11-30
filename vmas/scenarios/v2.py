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

    # def observation(self, agent: Agent):
    #     # Agent's position, velocity, and capacity
    #     pos = agent.state.pos  # [batch_dim, 2]
    #     vel = agent.state.vel  # [batch_dim, 2]
    #     capacity = torch.full((self.world.batch_dim, 1), agent.u_multiplier, device=self.world.device)

    #     obs = [pos, vel, capacity]  # Initial observations: 5 features

    #     # Lidar readings
    #     if self.use_lidar:
    #         lidar_measurements = agent.sensors[0].measure()  # [batch_dim, n_lidar_rays]
    #         obs.append(lidar_measurements)  # Add lidar readings to obs

    #     # Package information
    #     for package in self.packages:
    #         pos_to_goal = package.state.pos - package.goal.state.pos  # Vector from package to goal
    #         pos_to_agent = package.state.pos - agent.state.pos        # Vector from package to agent
    #         package_vel = package.state.vel                           # Package velocity
    #         on_goal = package.on_goal.unsqueeze(-1)                   # Goal status (boolean as a feature)

    #         obs.extend([pos_to_goal, pos_to_agent, package_vel, on_goal])  # 8 features per package

    #     return torch.cat(obs, dim=-1)  # Final concatenated observation tensor
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

        # Add agent's relative position to goal
        for package in self.packages:
            pos_to_goal = package.state.pos - package.goal.state.pos
            obs.append(pos_to_goal)  # Add to observation

        # Add a global goal indicator for the agent
        all_packages_on_goal = torch.all(torch.stack([package.on_goal for package in self.packages], dim=1), dim=-1)
        goal_indicator = all_packages_on_goal.unsqueeze(-1).float()
        obs.append(goal_indicator)

        # Add previous action taken by the agent (this assumes action history is tracked)
        if hasattr(agent, 'last_action'):
            obs.append(agent.last_action)  # Add the agent's previous action to the observation

        return torch.cat(obs, dim=-1)  # Final concatenated observation tensor


    # def reward(self, agent: Agent):
    #     # Only compute reward once per timestep, based on the first agent in list
    #     is_first = agent == self.world.agents[0]
    #     if is_first:
    #         self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

    #         # Reward for each package
    #         for package in self.packages:
    #             # Calculate distance to goal
    #             package.dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
    #             package.on_goal = self.world.is_overlapping(package, package.goal)
    #             package.color = torch.tensor(
    #                 Color.RED.value, device=self.world.device, dtype=torch.float32
    #             ).repeat(self.world.batch_dim, 1)
    #             package.color[package.on_goal] = torch.tensor(
    #                 Color.GREEN.value, device=self.world.device, dtype=torch.float32
    #             )

    #             # Shaping and reward based on reaching goal
    #             package_shaping = package.dist_to_goal * self.shaping_factor
    #             self.rew[~package.on_goal] += (
    #                 package.global_shaping[~package.on_goal] - package_shaping[~package.on_goal]
    #             )
    #             package.global_shaping = package_shaping

    #         # Energy penalty for each agent's movement
    #         self.energy_rew = self.energy_coeff * -torch.stack(
    #             [
    #                 torch.linalg.vector_norm(a.action.u, dim=-1)
    #                 / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
    #                 for a in self.world.agents
    #             ],
    #             dim=1,
    #         ).sum(-1)
    #         self.rew += self.energy_rew

    #     return self.rew  # Return reward tensor
    # def reward(self, agent: Agent):
        # # Only compute reward once per timestep, based on the first agent in the list
        # is_first = agent == self.world.agents[0]
        # if is_first:
        #     self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

        #     # Initialization for rewards
        #     goal_bonus = 100.0  # Bonus for reaching the goal
        #     collision_penalty = -1.0  # Penalty for collisions
        #     proximity_reward_coeff = 10.0  # Reward coefficient for staying near a package
        #     team_contribution_coeff = 50.0  # Reward coefficient for moving packages closer to the goal
        #     step_penalty = -1  # Penalty for each timestep taken by agents to encourage faster completion

        #     for package in self.packages:
        #         # Calculate distance to goal
        #         package.dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
        #         package.on_goal = self.world.is_overlapping(package, package.goal)

        #         # Color packages based on their goal status
        #         package.color = torch.tensor(
        #             Color.RED.value, device=self.world.device, dtype=torch.float32
        #         ).repeat(self.world.batch_dim, 1)
        #         package.color[package.on_goal] = torch.tensor(
        #             Color.GREEN.value, device=self.world.device, dtype=torch.float32
        #         )

        #         # **Step 1: Reward for Agents Getting Close to the Package**
        #         # Proximity reward for agents getting near to the package
        #         agents_near_package = torch.zeros(self.world.batch_dim, device=self.world.device)

        #         for agent in self.world.agents:
        #             agent_to_package_dist = torch.linalg.vector_norm(agent.state.pos - package.state.pos, dim=1)
        #             proximity_reward = proximity_reward_coeff * torch.exp(-agent_to_package_dist)
        #             self.rew += proximity_reward

        #             # Count agents near the package (within a certain threshold)
        #             agents_near_threshold = 0.2  # Distance threshold to consider agents near the package
        #             agents_near_package += (agent_to_package_dist < agents_near_threshold).float()

        #         # **Step 2: Reward for Moving the Package Towards the Goal**
        #         # Create a mask to check if enough agents are near the package for each batch
        #         min_agents_near_package = 2  # Define how many agents need to be near
        #         sufficient_agents_near = agents_near_package >= min_agents_near_package  # Shape: [batch_dim]

        #         # Only apply reward for moving the package if sufficient agents are near it
        #         if sufficient_agents_near.any():
        #             shaping_factor = torch.exp(-package.dist_to_goal) * self.shaping_factor
        #             reward_delta = package.global_shaping - shaping_factor

        #             # Apply reward only to those batches where sufficient agents are near the package
        #             self.rew[sufficient_agents_near] += reward_delta[sufficient_agents_near]

        #             # Update global shaping for all packages
        #             package.global_shaping = shaping_factor

        #             # Bonus reward for reaching the goal
        #             self.rew += goal_bonus * package.on_goal.float()

        #     # **Energy Penalty for Each Agent's Movement**
        #     self.energy_rew = self.energy_coeff * -torch.stack(
        #         [
        #             torch.linalg.vector_norm(a.action.u, dim=-1)
        #             / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
        #             for a in self.world.agents
        #         ],
        #         dim=1,
        #     ).sum(-1)
        #     self.rew += self.energy_rew

        #     # **Collision Penalty Between Agents**
            # collision_count = torch.zeros(self.world.batch_dim, device=self.world.device)
            # for agent in self.world.agents:
            #     for other_agent in self.world.agents:
            #         if agent != other_agent:
            #             is_collision = self.world.is_overlapping(agent, other_agent)
            #             if is_collision.any():
            #                 collision_count[is_collision] += 1

            # # Apply a penalty that scales with the number of collisions
            # self.rew += collision_penalty * collision_count

        #     # **Step Penalty for Each Step Taken**
        #     self.rew += step_penalty

        # return self.rew  # Return reward tensor
    # def reward(self, agent: Agent):
    #     is_first = agent == self.world.agents[0]
    #     if is_first:
    #         self.rew = torch.zeros(self.world.batch_dim, device=self.world.device, dtype=torch.float32)

    #         # Define constants
    #         goal_bonus = 100.0
    #         collision_penalty = -1.0
    #         proximity_reward_coeff = 10.0
    #         team_contribution_coeff = 50.0
    #         step_penalty = -1
    #         zone_radius = 0.5  # Define the zone radius around the package
    #         outside_zone_penalty = -5.0  # Penalty for agents outside the zone

    #         for package in self.packages:
    #             # Calculate distance to goal
    #             package.dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
    #             package.on_goal = self.world.is_overlapping(package, package.goal)

    #             # Color packages based on goal status
    #             package.color = torch.tensor(Color.RED.value, device=self.world.device, dtype=torch.float32).repeat(self.world.batch_dim, 1)
    #             package.color[package.on_goal] = torch.tensor(Color.GREEN.value, device=self.world.device, dtype=torch.float32)

    #             # Initialize counter for agents within the zone
    #             agents_in_zone = torch.zeros(self.world.batch_dim, device=self.world.device)

    #             for agent in self.world.agents:
    #                 agent_to_package_dist = torch.linalg.vector_norm(agent.state.pos - package.state.pos, dim=1)

    #                 # **Reward and Penalize Based on Zone**
    #                 # Determine if agent is within zone
    #                 in_zone = agent_to_package_dist < zone_radius
    #                 agents_in_zone += in_zone.float()

    #                 # Calculate contribution for agents within the zone
    #                 if in_zone.any():
    #                     # Reward based on motion towards the goal
    #                     contribution_reward = team_contribution_coeff * (1 - agent_to_package_dist / zone_radius)
    #                     self.rew[in_zone] += contribution_reward[in_zone]
    #                 else:
    #                     # Penalize agents outside the zone
    #                     self.rew += outside_zone_penalty

    #             # Goal bonus and step penalty
    #             self.rew += goal_bonus * package.on_goal.float()
    #             self.rew += step_penalty

    #     return self.rew

    def reward(self, agent: Agent):
    # Initialize per-agent rewards if not already done
        if not hasattr(self, 'per_agent_rew'):
            self.per_agent_rew = torch.zeros(
                (self.world.batch_dim, len(self.world.agents)), 
                device=self.world.device, 
                dtype=torch.float32
            )
        
        # Compute rewards only once per timestep
        is_first = agent == self.world.agents[0]
        if is_first:
            # Reset per-agent rewards
            self.per_agent_rew.zero_()
            
            # Define constants
            goal_bonus = 100.0
            collision_penalty = -1.0
            proximity_reward_coeff = 10.0
            team_contribution_coeff = 50.0
            step_penalty = -1
            zone_radius = 0.5
            outside_zone_penalty = -5.0

            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                
                # Color packages based on goal status
                package.color = torch.tensor(
                    Color.RED.value, device=self.world.device, dtype=torch.float32
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value, device=self.world.device, dtype=torch.float32
                )
                
                for idx, agent in enumerate(self.world.agents):
                    agent_to_package_dist = torch.linalg.vector_norm(
                        agent.state.pos - package.state.pos, dim=1
                    )

                    in_zone = agent_to_package_dist < zone_radius
                    # Reward for agents within the zone
                    contribution_reward = torch.zeros(self.world.batch_dim, device=self.world.device)
                    contribution_reward[in_zone] = team_contribution_coeff * (
                        1 - agent_to_package_dist[in_zone] / zone_radius
                    )
                    self.per_agent_rew[:, idx] += contribution_reward

                    # Penalty for agents outside the zone
                    self.per_agent_rew[:, idx] += (~in_zone).float() * outside_zone_penalty

                # Add goal bonus to all agents if package is on goal
                if package.on_goal.any():
                    self.per_agent_rew += goal_bonus * package.on_goal.float().unsqueeze(-1)
                
                # Step penalty for all agents
                self.per_agent_rew += step_penalty

        # Return per-agent rewards
        agent_idx = self.world.agents.index(agent)
        return self.per_agent_rew[:, agent_idx]



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
