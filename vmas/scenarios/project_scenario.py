'''
import torch
import math
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.joints import Joint
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
    # def create_joint(self, entity_a, entity_b, distance):
    #     """
    #     Creates a joint between two entities (e.g., agent and package).
        
    #     Args:
    #         entity_a: The first entity (e.g., an agent).
    #         entity_b: The second entity (e.g., a package).
    #         distance: The initial distance for the joint.
    #     """
    #     # Create the joint object (assuming you have a joint class or API in the environment)
    #     joint = Joint(entity_a=entity_a, entity_b=entity_b, distance=distance)
        
    #     # Add the joint to the world's list of joints
    #     self.world.joints.append(joint)
    #     print(f"Joint created between {entity_a.name} and {entity_b.name} with distance {distance}")
    # def create_joint(self, agent: Agent, package: Landmark, distance):
    #     joint = Joint(
    #         entity_a=agent,
    #         entity_b=package,
    #         anchor_a=(0.0, 0.0),
    #         anchor_b=(0.0, 0.0),
    #         rotate_a=False,
    #         rotate_b=False,
    #         dist=distance,
    #         collidable=False,
    #         width=0.0,
    #         mass=1.0,
    #     )
    #     self.world.add_joint(joint)
    #     print(f"Joint created between {agent} and {package} with initial distance {distance}")
    def create_joint(self, agent: Agent, package: Landmark, distance):
    # Introducing a gradually reducing distance for smooth attachment
        target_distance = 0.1  # Target distance for smooth attachment
        smooth_dist = max(target_distance, distance * 0.9)  # Smoothly reduce distance by 10% per step, but not below target

        # Create the joint with smoother properties
        joint = Joint(
            entity_a=agent,
            entity_b=package,
            anchor_a=(0.0, 0.0),  # Adjust anchor if needed for alignment
            anchor_b=(0.0, 0.0),
            rotate_a=False,
            rotate_b=False,
            dist=smooth_dist,
            collidable=False,
            width=0.05,  # Small width for flexibility
            mass=0.5,  # Moderate mass for smooth response
        )

        # Add the joint to the world and provide feedback for debugging
        self.world.add_joint(joint)
        print(f"Smooth joint created between {agent} and {package} with initial smooth distance {smooth_dist}")




    # def reward(self, agent: Agent):
    #     is_first = agent == self.world.agents[0]

    #     if is_first:
    #         self.rew = torch.zeros(
    #             self.world.batch_dim,
    #             device=self.world.device,
    #             dtype=torch.float32,
    #         )

    #         for package in self.packages:
    #             package.dist_to_goal = torch.linalg.vector_norm(
    #                 package.state.pos - package.goal.state.pos, dim=1
    #             )

    #             package.on_goal = self.world.is_overlapping(package, package.goal)
    #             package.color = torch.tensor(
    #                 Color.RED.value,
    #                 device=self.world.device,
    #                 dtype=torch.float32,
    #             ).repeat(self.world.batch_dim, 1)
    #             package.color[package.on_goal] = torch.tensor(
    #                 Color.GREEN.value,
    #                 device=self.world.device,
    #                 dtype=torch.float32,
    #             )

    #             package_shaping = package.dist_to_goal * self.shaping_factor
    #             self.rew += (
    #                 package.global_shaping - package_shaping
    #             )
    #             package.global_shaping = package_shaping

    #         # Energy penalty
    #         self.energy_rew = self.energy_coeff * -torch.stack(
    #             [
    #                 torch.linalg.vector_norm(a.action.u, dim=-1)
    #                 / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
    #                 for a in self.world.agents
    #             ],
    #             dim=1,
    #         ).sum(-1)

    #         # Total reward
    #         self.rew += self.energy_rew

    #     return self.rew
    # def reward(self, agent: Agent):
    #     is_first = agent == self.world.agents[0]

    #     if is_first:
    #         self.rew = torch.zeros(
    #             self.world.batch_dim,
    #             device=self.world.device,
    #             dtype=torch.float32,
    #         )

    #         # Compute distance-based reward for all packages
    #         total_package_reward = 0
    #         for package in self.packages:
    #             package.dist_to_goal = torch.linalg.vector_norm(
    #                 package.state.pos - package.goal.state.pos, dim=1
    #             )

    #             # Check if package is on the goal
    #             package.on_goal = self.world.is_overlapping(package, package.goal)

    #             # Assign colors based on whether the package is on the goal
    #             package.color = torch.where(
    #                 package.on_goal.unsqueeze(-1),  # Expand for color assignment if needed
    #                 torch.tensor(Color.GREEN.value, device=self.world.device, dtype=torch.float32),
    #                 torch.tensor(Color.RED.value, device=self.world.device, dtype=torch.float32)
    #             )

    #             # Reward shaping: reward reduction in distance to goal
    #             package_shaping = package.dist_to_goal * self.shaping_factor
    #             total_package_reward += (package.global_shaping - package_shaping).sum()
    #             package.global_shaping = package_shaping

    #         # Apply global reward based on package progress
    #         self.rew += total_package_reward / len(self.packages)  # Normalize by number of packages

    #         # Compute energy penalty for all agents
    #         self.energy_rew = self.energy_coeff * -torch.stack(
    #             [
    #                 torch.linalg.vector_norm(a.action.u, dim=-1)
    #                 / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
    #                 for a in self.world.agents
    #             ],
    #             dim=1,
    #         ).sum(-1)

    #         # Add energy penalty to reward
    #         self.rew += self.energy_rew

    #     return self.rew
    # def reward(self, agent: Agent):
    #     is_first = agent == self.world.agents[0]

    #     if is_first:
    #         # Initialize reward
    #         self.rew = torch.zeros(
    #             self.world.batch_dim,
    #             device=self.world.device,
    #             dtype=torch.float32,
    #         )

    #         # Compute package progress reward
    #         total_package_reward = 0
    #         for package in self.packages:
    #             # Calculate distance to goal
    #             package.dist_to_goal = torch.linalg.vector_norm(
    #                 package.state.pos - package.goal.state.pos, dim=1
    #             )
    #             # Check if package reached goal
    #             package.on_goal = self.world.is_overlapping(package, package.goal)

    #             # Set package color based on status
    #             package.color = torch.where(
    #                 package.on_goal.unsqueeze(-1),
    #                 torch.tensor(Color.GREEN.value, device=self.world.device, dtype=torch.float32),
    #                 torch.tensor(Color.RED.value, device=self.world.device, dtype=torch.float32)
    #             )

    #             # Reward shaping: reward progress towards the goal
    #             package_shaping = package.dist_to_goal * self.shaping_factor
    #             total_package_reward += (package.global_shaping - package_shaping).sum()
    #             package.global_shaping = package_shaping

    #         # Average reward based on package progress
    #         self.rew += total_package_reward / len(self.packages)

    #         # Agent movement incentives
    #         total_agent_reward = 0
    #         for agent in self.world.agents:
    #             # Initialize `global_shaping` for each agent if it doesn’t exist
    #             if not hasattr(agent, 'global_shaping'):
    #                 agent.global_shaping = torch.zeros(self.world.batch_dim, device=self.world.device)

    #             agent_to_package_dist = torch.stack(
    #                 [torch.linalg.vector_norm(agent.state.pos - p.state.pos, dim=1) for p in self.packages]
    #             )
    #             min_distance = agent_to_package_dist.min(dim=0)[0]  # Minimum distance to any package

    #             # Encourage agents to get closer to packages
    #             agent_shaping = min_distance * self.shaping_factor
    #             total_agent_reward += (agent.global_shaping - agent_shaping).sum()
    #             agent.global_shaping = agent_shaping

    #             # Penalize for moving without meaningful progress
    #             movement_penalty = self.energy_coeff * torch.linalg.vector_norm(agent.state.vel, dim=-1)
    #             self.rew -= movement_penalty.sum()

    #         # Total agent reward normalized by the number of agents
    #         self.rew += total_agent_reward / len(self.world.agents)

    #         # Energy penalty across agents for unnecessary movement
    #         self.energy_rew = self.energy_coeff * -torch.stack(
    #             [
    #                 torch.linalg.vector_norm(a.action.u, dim=-1)
    #                 / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
    #                 for a in self.world.agents
    #             ],
    #             dim=1,
    #         ).sum(-1)

    #         # Add energy penalty to reward
    #         self.rew += self.energy_rew

    #     return self.rew




    # def observation(self, agent: Agent):
    #     package_obs = []
    #     for package in self.packages:
    #         package_obs.append(package.state.pos - package.goal.state.pos)  # Package to goal
    #         package_obs.append(package.state.pos - agent.state.pos)         # Package to agent
    #         package_obs.append(package.state.vel)                           # Package velocity
    #         package_obs.append(package.on_goal.unsqueeze(-1).float())       # Package on goal

    #     return torch.cat(
    #         [
    #             agent.state.pos,     # Agent position
    #             agent.state.vel,     # Agent velocity
    #             *package_obs,
    #         ],
    #         dim=-1,
    #     )

    # def done(self):
    #     return torch.all(
    #         torch.stack(
    #             [package.on_goal for package in self.packages],
    #             dim=1,
    #         ),
    #         dim=-1,
    #     )
    
    # def pre_step(self, agents):
    #     # Adjusted `pre_step` method to handle `agents` as an argument
    #     threshold_distance = 0.25  # Example threshold for proximity-based joint creation

    #     for agent in agents:
    #         for package in self.packages:
    #             distance = torch.dist(agent.state.pos, package.state.pos)
    #             if distance < threshold_distance and torch.all(agent.state.vel == 0):
    #                 print(f"Distance between {agent} and {package} is {distance.item()}")
    #                 self.create_joint(agent, package, distance)

    #     # Call the base class pre_step (without arguments)
    #     super().pre_step()
    # def reward(self, agent: Agent):
    #     is_first = agent == self.world.agents[0]

    #     if is_first:
    #         # Initialize reward
    #         self.rew = torch.zeros(
    #             self.world.batch_dim,
    #             device=self.world.device,
    #             dtype=torch.float32,
    #         )

    #         # Compute package progress reward
    #         total_package_reward = 0
    #         for package in self.packages:
    #             # Calculate distance to goal
    #             package.dist_to_goal = torch.linalg.vector_norm(
    #                 package.state.pos - package.goal.state.pos, dim=1
    #             )
    #             # Check if package reached goal
    #             package.on_goal = self.world.is_overlapping(package, package.goal)

    #             # Reward shaping: reward progress towards the goal
    #             package_shaping = package.dist_to_goal * self.shaping_factor
    #             total_package_reward += (package.global_shaping - package_shaping).sum()
    #             package.global_shaping = package_shaping

    #         # Average reward based on package progress
    #         self.rew += total_package_reward / len(self.packages)

    #         # Agent movement incentives and penalties
    #         total_agent_reward = 0
    #         for agent in self.world.agents:
    #             if not hasattr(agent, 'global_shaping'):
    #                 agent.global_shaping = torch.zeros(self.world.batch_dim, device=self.world.device)

    #             # Distance to nearest package
    #             agent_to_package_dist = torch.stack(
    #                 [torch.linalg.vector_norm(agent.state.pos - p.state.pos, dim=1) for p in self.packages]
    #             )
    #             min_distance = agent_to_package_dist.min(dim=0)[0]

    #             # Shaping based on proximity to the nearest package
    #             agent_shaping = min_distance * self.shaping_factor
    #             total_agent_reward += (agent.global_shaping - agent_shaping).sum()
    #             agent.global_shaping = agent_shaping

    #             # Penalize for unnecessary movement when there's no progress
    #             movement_penalty = self.energy_coeff * torch.linalg.vector_norm(agent.state.vel, dim=-1)
    #             if torch.all(agent_shaping >= agent.global_shaping):
    #                 self.rew -= movement_penalty.sum()

    #         # Normalize the total agent reward by the number of agents
    #         self.rew += total_agent_reward / len(self.world.agents)

    #         # Energy penalty for all agents' actions
    #         self.energy_rew = self.energy_coeff * -torch.stack(
    #             [
    #                 torch.linalg.vector_norm(a.action.u, dim=-1)
    #                 / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
    #                 for a in self.world.agents
    #             ],
    #             dim=1,
    #         ).sum(-1)

    #         # Add energy penalty to reward
    #         self.rew += self.energy_rew

    #     return self.rew



    # def observation(self, agent: Agent):
    #     package_obs = []
    #     for package in self.packages:
    #         package_obs.append(package.state.pos - package.goal.state.pos)  # Package to goal
    #         package_obs.append(package.state.pos - agent.state.pos)         # Package to agent
    #         package_obs.append(package.state.vel)                           # Package velocity
    #         package_obs.append(package.on_goal.unsqueeze(-1).float())       # Package on goal

    #     return torch.cat(
    #         [
    #             agent.state.pos,     # Agent position
    #             agent.state.vel,     # Agent velocity
    #             *package_obs,
    #         ],
    #         dim=-1,
    #     )


    # def done(self):
    #     return torch.all(
    #         torch.stack(
    #             [package.on_goal for package in self.packages],
    #             dim=1,
    #         ),
    #         dim=-1,
    #     )


    # def pre_step(self, agents):
    #     # Adjusted `pre_step` method to handle `agents` as an argument
    #     threshold_distance = 0.25  # Example threshold for proximity-based joint creation

    #     for agent in agents:
    #         for package in self.packages:
    #             distance = torch.dist(agent.state.pos, package.state.pos)
    #             if distance < threshold_distance and torch.all(agent.state.vel == 0):
    #                 print(f"Distance between {agent} and {package} is {distance.item()}")
    #                 self.create_joint(agent, package, distance)

    #     # Call the base class pre_step (without arguments)
    #     super().pre_step()

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            # Initialize reward
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )

            # Compute package progress reward
            total_package_reward = 0
            for package in self.packages:
                # Initialize package.global_shaping if not present
                if not hasattr(package, 'global_shaping'):
                    package.global_shaping = torch.zeros(self.world.batch_dim, device=self.world.device)

                # Store previous shaping
                prev_package_shaping = package.global_shaping

                # Calculate distance to goal
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                # Check if package reached goal
                package.on_goal = self.world.is_overlapping(package, package.goal)

                # Reward shaping: reward progress towards the goal
                package_shaping = package.dist_to_goal * self.shaping_factor
                shaping_diff = prev_package_shaping - package_shaping
                total_package_reward += shaping_diff.sum()
                package.global_shaping = package_shaping

            # Average reward based on package progress
            self.rew += total_package_reward / len(self.packages)

            # Agent movement incentives and penalties
            total_agent_reward = 0
            for agent in self.world.agents:
                if not hasattr(agent, 'global_shaping'):
                    agent.global_shaping = torch.zeros(self.world.batch_dim, device=self.world.device)

                # Store previous shaping
                prev_agent_shaping = agent.global_shaping

                # Distance to nearest package
                agent_to_package_dist = torch.stack(
                    [torch.linalg.vector_norm(agent.state.pos - p.state.pos, dim=1) for p in self.packages]
                )
                min_distance = agent_to_package_dist.min(dim=0)[0]

                # Shaping based on proximity to the nearest package
                agent_shaping = min_distance * self.shaping_factor
                shaping_diff = prev_agent_shaping - agent_shaping
                total_agent_reward += shaping_diff.sum()
                agent.global_shaping = agent_shaping

                # Penalize for unnecessary movement when there's no progress
                movement_penalty = self.energy_coeff * torch.linalg.vector_norm(agent.state.vel, dim=-1)
                if torch.all(shaping_diff <= 0):  # No progress made
                    self.rew -= movement_penalty.sum()

            # Normalize the total agent reward by the number of agents
            self.rew += total_agent_reward / len(self.world.agents)

            # Energy penalty for all agents' actions
            self.energy_rew = self.energy_coeff * -torch.stack(
                [
                    torch.linalg.vector_norm(a.action.u, dim=-1)
                    / math.sqrt(self.world.dim_p * ((a.u_range * a.u_multiplier) ** 2))
                    for a in self.world.agents
                ],
                dim=1,
            ).sum(-1)

            # Add energy penalty to reward
            self.rew += self.energy_rew

        return self.rew
    
    # def observation(self, agent: Agent):
    #     package_obs = []
    #     for package in self.packages:
    #         package_obs.append(package.state.pos - package.goal.state.pos)  # Package to goal
    #         package_obs.append(package.state.pos - agent.state.pos)         # Package to agent
    #         package_obs.append(package.state.vel)                           # Package velocity
    #         package_obs.append(package.on_goal.unsqueeze(-1).float())       # Package on goal

    #     # Include observations about other agents
    #     other_agents_obs = []
    #     for other_agent in self.world.agents:
    #         if other_agent != agent:
    #             other_agents_obs.append(other_agent.state.pos - agent.state.pos)  # Other agent's position relative to this agent
    #             other_agents_obs.append(other_agent.state.vel)                    # Other agent's velocity

    #     return torch.cat(
    #         [
    #             agent.state.pos,     # Agent position
    #             agent.state.vel,     # Agent velocity
    #             *package_obs,        # Package observations
    #             *other_agents_obs,   # Other agents' observations
    #         ],
    #         dim=-1,
    #     )

    def observation(self, agent: Agent):
        package_obs = []
        for package in self.packages:
            # Direction from agent to package's goal
            dir_to_goal = package.goal.state.pos - agent.state.pos
            dir_to_goal_norm = dir_to_goal / (torch.linalg.norm(dir_to_goal, dim=-1, keepdim=True) + 1e-6)

            # Direction from agent to package
            dir_to_package = package.state.pos - agent.state.pos
            dir_to_package_norm = dir_to_package / (torch.linalg.norm(dir_to_package, dim=-1, keepdim=True) + 1e-6)

            package_obs.append(dir_to_goal_norm)                       # Normalized direction to package goal
            package_obs.append(dir_to_package_norm)                    # Normalized direction to package
            package_obs.append(package.state.vel)                      # Package velocity
            package_obs.append(package.on_goal.unsqueeze(-1).float())  # Package on goal

        # Include observations about other agents
        other_agents_obs = []
        for other_agent in self.world.agents:
            if other_agent != agent:
                # Direction from agent to other agent
                dir_to_other_agent = other_agent.state.pos - agent.state.pos
                dir_to_other_agent_norm = dir_to_other_agent / (torch.linalg.norm(dir_to_other_agent, dim=-1, keepdim=True) + 1e-6)

                other_agents_obs.append(dir_to_other_agent_norm)  # Normalized direction to other agent
                other_agents_obs.append(other_agent.state.vel)    # Other agent's velocity

        # Include agent's own heading (assuming agent's velocity can represent heading)
        agent_heading = agent.state.vel / (torch.linalg.norm(agent.state.vel, dim=-1, keepdim=True) + 1e-6)

        return torch.cat(
            [
                agent.state.pos,             # Agent position
                agent.state.vel,             # Agent velocity
                agent_heading,               # Agent's own heading
                *package_obs,                # Package observations
                *other_agents_obs,           # Other agents' observations
            ],
            dim=-1,
        )

    
    # def pre_step(self, agents):
    #     threshold_distance = 0.25  # Distance threshold for creating a joint
    #     velocity_threshold = 0.01  # Velocity threshold to check if the agent is stationary

    #     for batch_index in range(self.world.batch_dim):  # Iterate over each batch environment
    #         for agent in agents:
    #             agent_speed = torch.linalg.norm(agent.state.vel[batch_index])  # Get speed of the agent in the current batch

    #             for package in self.packages:
    #                 # Calculate the distance between the agent and package for the current batch
    #                 distance = torch.linalg.norm(agent.state.pos[batch_index] - package.state.pos[batch_index])

    #                 # Print debugging information about distance and speed conditions
    #                 # print(f"Batch {batch_index} | Agent: {agent} | Package: {package}")
    #                 # print(f"Distance: {distance} | Speed: {agent_speed}")

    #                 # Check if the agent is close to the package and stationary
    #                 if distance < threshold_distance and agent_speed < velocity_threshold:
    #                     # Ensure the joint hasn't already been created
    #                     if not any(joint.entity_a == agent and joint.entity_b == package for joint in self.world.joints):
    #                         print(f"Creating joint between {agent} and {package} in batch {batch_index} with distance {distance}")
    #                         # Create joint
    #                         self.create_joint(agent, package, distance.item())
    #                     else:
    #                         print(f"Joint already exists between {agent} and {package} in batch {batch_index}")
    def pre_step(self, agents):
    # Update distance of each joint smoothly
        for joint in self.world.joints:
            if joint.entity_a in agents and joint.entity_b in self.packages:
                # Smoothly adjust distance
                current_dist = joint.dist
                target_distance = 0.1
                joint.dist = max(target_distance, current_dist * 0.9)


    def done(self):
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )
'''

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
        self.max_episode_length = 1000
        self.current_step = torch.zeros(batch_dim, device=device, dtype=torch.int32)

        # self.world = world
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

    def create_joint(self, agent: Agent, package: Landmark, distance):
        # Ensure a minimum distance to prevent None values
        distance = max(distance, 0.1)  # Set a reasonable minimum distance if distance is too small

        joint = Joint(
            entity_a=agent,
            entity_b=package,
            anchor_a=(0.0, 0.0),       # Set anchor points to fixed values
            anchor_b=(0.0, 0.0),
            rotate_a=False,            # Set rotation to False as needed
            rotate_b=False,
            dist=distance,             # Use the adjusted distance
            collidable=True,           # Set collidable to True to avoid AssertionError
            width=0.05,                # Set width to avoid None values
            mass=0.5                   # Set mass to avoid None values
        )
        self.world.add_joint(joint)
        print(f"Joint created between {agent.name} and {package.name} with initial distance {distance}")




    def pre_step(self, agents):
        threshold_distance = 0.25  # Distance threshold for creating a joint
        velocity_threshold = 0.01  # Velocity threshold to check if the agent is stationary

        for agent in agents:
            agent_speed = torch.linalg.norm(agent.state.vel, dim=-1)  # Calculate agent speed

            for package in self.packages:
                # Calculate the distance between the agent and package for each batch
                distance = torch.linalg.norm(agent.state.pos - package.state.pos, dim=-1)  # Shape: (batch_dim,)

                for batch_idx in range(distance.shape[0]):  # Check each batch separately
                    if (distance[batch_idx] < threshold_distance) and (agent_speed[batch_idx] < velocity_threshold):
                        # Check if a joint already exists between the agent and package in the specific batch
                        if not any(joint.entity_a == agent and joint.entity_b == package for joint in self.world.joints):
                            self.create_joint(agent, package, max(distance[batch_idx].item(), 0.1))  # Set minimum distance
                            print(f"Joint created in batch {batch_idx} between {agent.name} and {package.name} with distance {distance[batch_idx].item()}")

        # Smoothly adjust distances of existing joints
        for joint in self.world.joints:
            if joint.entity_a in agents and joint.entity_b in self.packages:
                current_dist = joint.dist
                target_distance = 0.1  # Minimum target distance
                joint.dist = max(target_distance, current_dist * 0.9)

        # Increment step counter for each environment in the batch
        self.current_step += 1



    def reward(self, agent: Agent):
        # Initialize agent reward
        agent_reward = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.float32,
        )

        # Check if agent is interacting with a package
        interacting = any(
            joint.entity_a == agent and joint.entity_b in self.packages
            for joint in self.world.joints
        )

        if interacting:
            # Agent is interacting with a package
            for package in self.packages:
                # Check if agent is attached to this package
                if any(
                    joint.entity_a == agent and joint.entity_b == package
                    for joint in self.world.joints
                ):
                    # Compute reward based on package movement
                    if not hasattr(package, 'global_shaping'):
                        package.global_shaping = torch.zeros(
                            self.world.batch_dim, device=self.world.device
                        )

                    prev_package_shaping = package.global_shaping

                    # Calculate distance to goal
                    package.dist_to_goal = torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    # Check if package reached goal
                    package.on_goal = self.world.is_overlapping(package, package.goal)

                    # Reward shaping: reward progress towards the goal
                    package_shaping = package.dist_to_goal * self.shaping_factor
                    shaping_diff = prev_package_shaping - package_shaping
                    agent_reward += shaping_diff
                    package.global_shaping = package_shaping
        else:
            # Penalize agent for not interacting
            agent_reward -= 0.1  # Penalty value for idleness

        # Energy penalty for the agent
        energy_penalty = self.energy_coeff * -torch.linalg.vector_norm(
            agent.action.u, dim=-1
        ) / math.sqrt(self.world.dim_p * ((agent.u_range * agent.u_multiplier) ** 2))
        agent_reward += energy_penalty

        return agent_reward

    def observation(self, agent: Agent):
        # Check if agent is attached to a package
        interacting = any(
            joint.entity_a == agent and joint.entity_b in self.packages
            for joint in self.world.joints
        )
        interaction_flag = torch.tensor(
            [float(interacting)], device=self.world.device
        ).repeat(self.world.batch_dim, 1)

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
                    dir_to_goal_norm,                      # Normalized direction to package goal
                    dir_to_package_norm,                   # Normalized direction to package
                    package.state.vel,                     # Package velocity
                    package.on_goal.unsqueeze(-1).float(), # Package on goal
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
                        other_agent.state.vel,    # Other agent's velocity
                    ]
                )

        # Include agent's own heading
        agent_heading = agent.state.vel / (
            torch.linalg.norm(agent.state.vel, dim=-1, keepdim=True) + 1e-6
        )

        return torch.cat(
            [
                agent.state.pos,    # Agent position
                agent.state.vel,    # Agent velocity
                agent_heading,      # Agent's own heading
                interaction_flag,   # Interaction flag
                *package_obs,       # Package observations
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


