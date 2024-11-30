# scenario_block_pushing.py

import torch
from vmas.simulator.core import Agent, Landmark, World
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

class Scenario(BaseScenario):
    def __init__(self, num_agents=2):
        self.num_agents = num_agents

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=0.1,
            collision_force=1.0,
            contact_margin=0.01,
        )
        world.dim_c = 0  # No communication

        # Create agents
        world.agents = [Agent(name=f"agent_{i}") for i in range(self.num_agents)]
        for agent in world.agents:
            agent.size = 0.05
            agent.accel = 3.0
            agent.max_speed = 1.0
            agent.color = Color(0.35, 0.35, 0.85)

        # Create movable block (landmark)
        self.block = Landmark(name="block", movable=True)
        self.block.size = 0.15
        self.block.initial_mass = 5.0  # Heavier than agents
        self.block.color = Color(0.25, 0.25, 0.25)
        world.landmarks = [self.block]

        # Create goal landmark (not movable)
        self.goal = Landmark(name="goal", movable=False)
        self.goal.size = 0.1
        self.goal.color = Color(0.15, 0.65, 0.15)
        world.landmarks.append(self.goal)

        # Set world properties
        self.world = world
        self.world_radius = 1.0

        return world

    def reset_world_at(self, env_index: int = None):
        # Randomize agent positions
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            min_dist_between_entities=0.1,
            x_bounds=(-self.world_radius, self.world_radius),
            y_bounds=(-self.world_radius, self.world_radius),
            env_index=env_index,
        )

        # Randomize block position
        ScenarioUtils.spawn_entities_randomly(
            [self.block],
            self.world,
            min_dist_between_entities=0.1,
            x_bounds=(-self.world_radius / 2, self.world_radius / 2),
            y_bounds=(-self.world_radius / 2, self.world_radius / 2),
            env_index=env_index,
        )

        # Randomize goal position
        ScenarioUtils.spawn_entities_randomly(
            [self.goal],
            self.world,
            min_dist_between_entities=0.1,
            x_bounds=(-self.world_radius, self.world_radius),
            y_bounds=(-self.world_radius, self.world_radius),
            env_index=env_index,
        )

    def reward(self, agent: Agent):
        # Negative distance from block to goal as reward
        dist_to_goal = torch.linalg.norm(
            self.block.state.pos - self.goal.state.pos, dim=-1
        )
        reward = -dist_to_goal

        # Encourage agents to push the block
        dist_agent_to_block = torch.linalg.norm(
            agent.state.pos - self.block.state.pos, dim=-1
        )
        reward -= dist_agent_to_block * 0.1  # Small penalty for being far from block

        return reward

    def observation(self, agent: Agent):
        # Agent's own position and velocity
        obs = [agent.state.pos, agent.state.vel]

        # Relative position and velocity of the block
        obs.append(self.block.state.pos - agent.state.pos)
        obs.append(self.block.state.vel)

        # Relative position of the goal to the block
        obs.append(self.goal.state.pos - self.block.state.pos)

        # Positions and velocities of other agents
        for other_agent in self.world.agents:
            if other_agent is not agent:
                obs.append(other_agent.state.pos - agent.state.pos)
                obs.append(other_agent.state.vel)

        # Concatenate observations
        return torch.cat(obs, dim=-1)

    def done(self):
        # Episode is done if the block reaches the goal
        dist_to_goal = torch.linalg.norm(
            self.block.state.pos - self.goal.state.pos, dim=-1
        )
        return (dist_to_goal < self.goal.size + self.block.size).to(self.world.device)
