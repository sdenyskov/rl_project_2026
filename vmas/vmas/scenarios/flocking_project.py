#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
from collections import deque
from typing import Callable, Dict

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        # Add inputable fields
        n_agents = kwargs.pop("n_agents", 4)
        n_obstacles = kwargs.pop("n_obstacles", 5)

        self.spawn_agents_together = kwargs.pop("spawn_agents_together", False)
        self.boundary_type = kwargs.pop("boundary_type", "none") # options: "none", "periodic", "clip", "walls"
        
        self.collision_reward = kwargs.pop("collision_reward", -0.1)
        self.dist_shaping_factor = kwargs.pop("dist_shaping_factor", 1)
        self.phi_reward_scale = kwargs.pop("phi_reward_scale", 0.05)

        self.phi_window = kwargs.pop("phi_window", 20)
        self.communication_range_dist = kwargs.pop("communication_range_dist", 1.0)
        self.communication_range_phi = kwargs.pop("communication_range_phi", 1.0)
        self.max_range_lidar = kwargs.pop("max_range_lidar", 0.2)
        self.localise_phi = kwargs.pop("localise_phi", False)

        self.obstacles_file = kwargs.pop("obstacles_file", None)
        self.agents_file = kwargs.pop("agents_file", None)
        self.destination_file = kwargs.pop("destination_file", None)

        ScenarioUtils.check_kwargs_consumed(kwargs)
        
        # Add internal fields
        self.plot_grid = True
        self.use_phi_instead_of_deltaphi = False
        
        self.n_lidar_rays = 12
        self.desired_distance = 0.1
        self.min_collision_distance = 0.005
        self._min_dist_between_entities = 0.1

        self.x_dim = 1
        self.y_dim = 1
        self.world_limit_x_dim = 1.5
        self.world_limit_y_dim = 1.5
        self.spawnzone_x_dim = 0.25
        self.spawnzone_y_dim = 0.25

        # Make world
        world = World(batch_dim, device, collision_force=400, substeps=5)

        # Add target agent
        self._target = Agent(
            name="target",
            collide=True,
            color=Color.GREEN,
            render_action=True,
            action_script=self.action_script_creator(),
        )
        world.add_agent(self._target)

        # Add agents
        goal_entity_filter: Callable[[Entity], bool] = lambda e: not isinstance(e, Agent)
        for i in range(n_agents):
            agent = Agent(
                name=f"agent_{i}",
                collide=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=self.n_lidar_rays,
                        max_range=self.max_range_lidar,
                        entity_filter=goal_entity_filter,
                    )
                ],
                render_action=True,
            )
            agent.collision_rew = torch.zeros(batch_dim, device=device)
            agent.dist_rew = agent.collision_rew.clone()
            world.add_agent(agent) 

        # Add walls
        self.walls = []
        if self.boundary_type == "walls":
            wall_specs = [
                ("wall_left", 0.1, self.world_limit_y_dim * 2),
                ("wall_right", 0.1, self.world_limit_y_dim * 2),
                ("wall_bottom", self.world_limit_x_dim * 2, 0.1),
                ("wall_top", self.world_limit_x_dim * 2, 0.1),
            ]
            for name, length, width in wall_specs:
                wall = Landmark(
                    name=name,
                    collide=True,
                    movable=False,
                    shape=Box(
                        length=length,
                        width=width,
                    ),
                    color=Color.GRAY,
                )
                world.add_landmark(wall)
                self.walls.append(wall)
        
        # Add obstacles
        self.obstacles = []
        for i in range(n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}",
                collide=True,
                movable=False,
                shape=Sphere(radius=0.1),
                color=Color.RED,
            )
            world.add_landmark(obstacle)
            self.obstacles.append(obstacle)
        
        # Add fields for computation of phi
        self.state_history = [deque(maxlen=self.phi_window) for _ in range(batch_dim)]
        shape = ((batch_dim, len(world.policy_agents)) if self.localise_phi else (batch_dim,))
        self.phi_reward = torch.zeros(shape, device=device)
        self.phi_prev = torch.zeros(shape, device=device)

        # Add fields for reward tracking
        self.reward_stats = {"collision": 0, "distance": 0, "phi": 0,}

        return world

    def action_script_creator(self):
        def action_script(agent, world):
            t = self.t / 30
            agent.action.u = torch.stack([torch.cos(t), torch.sin(t)], dim=1)

        return action_script

    def reset_world_at(self, env_index: int = None):

        # start with walls, if any
        if self.boundary_type == "walls":
            wall_positions = [
                (- self.world_limit_x_dim - 0.05, 0),  # left
                (self.world_limit_x_dim + 0.05, 0),    # right
                (0, - self.world_limit_y_dim - 0.05),  # bottom
                (0, self.world_limit_y_dim + 0.05),    # top
            ]
            for wall, pos in zip(self.walls, wall_positions):
                wall_pos = torch.tensor(pos, dtype=torch.float32, device=self.world.device,)
                wall_pos = wall_pos.expand(self.world.batch_dim, -1,)
                wall.set_pos(wall_pos, batch_index=env_index,)

        # target agent
        target_pos = torch.zeros(((1, self.world.dim_p) if env_index is not None else (self.world.batch_dim, self.world.dim_p)), device=self.world.device, dtype=torch.float32,)
        target_pos[:, Y] = -self.y_dim
        self._target.set_pos(target_pos, batch_index=env_index)

        # obstacles
        if self.obstacles_file is not None and self.obstacles_file != "":
            data = torch.load(self.obstacles_file, map_location=self.world.device,)
            if env_index is None:
                for i, obstacle in enumerate(self.obstacles):
                    obstacle.set_pos(data["obstacles"][:, i], batch_index=None,)
            else:
                for i, obstacle in enumerate(self.obstacles):
                    obstacle.state.pos[env_index] = (data["obstacles"][env_index, i])
        else:
            ScenarioUtils.spawn_entities_randomly(
                self.obstacles,
                self.world,
                env_index,
                self._min_dist_between_entities,
                x_bounds=(-self.x_dim, self.x_dim),
                y_bounds=(-self.y_dim, self.y_dim),
                occupied_positions=target_pos.unsqueeze(1),
            )
        
        # agents
        if self.agents_file is not None and self.agents_file != "":
            data = torch.load(self.agents_file, map_location=self.world.device,)
            if env_index is None:
                for i, agent in enumerate(self.world.policy_agents):
                    agent.set_pos(data["agents"][:, i], batch_index=None,)
            else:
                for i, agent in enumerate(self.world.policy_agents):
                    agent.state.pos[env_index] = (data["agents"][env_index, i])
        else:
            ScenarioUtils.spawn_entities_randomly(
                self.world.policy_agents,
                self.world,
                env_index,
                self._min_dist_between_entities,
                x_bounds=(-self.spawnzone_x_dim, self.spawnzone_x_dim) if self.spawn_agents_together else (-self.x_dim, self.x_dim),
                y_bounds=(-self.y_dim-self.spawnzone_y_dim, -self.y_dim+self.spawnzone_y_dim) if self.spawn_agents_together else (-self.y_dim, self.y_dim),
                occupied_positions=target_pos.unsqueeze(1),
            )           

        for agent in self.world.policy_agents:
            if env_index is None:
                agent.distance_shaping = (
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                agent.state.pos - a.state.pos, dim=-1
                            )
                            for a in self.world.agents
                            if a != agent
                        ],
                        dim=1,
                    )
                    - self.desired_distance
                ).pow(2).mean(-1) * self.dist_shaping_factor
            else:
                agent.distance_shaping[env_index] = (
                    torch.stack(
                        [
                            torch.linalg.vector_norm(
                                agent.state.pos[env_index] - a.state.pos[env_index]
                            )
                            for a in self.world.agents
                            if a != agent
                        ],
                        dim=0,
                    )
                    - self.desired_distance
                ).pow(2).mean(-1) * self.dist_shaping_factor

        if env_index is None:
            self.t = torch.zeros(self.world.batch_dim, device=self.world.device)
        else:
            self.t[env_index] = 0
        
        if self.destination_file is not None and self.destination_file != "" and env_index is None:
            data = {
                "agents": torch.stack([agent.state.pos.clone() for agent in self.world.policy_agents], dim=1,),
                "obstacles": torch.stack([obstacle.state.pos.clone() for obstacle in self.obstacles], dim=1,),
                "target": self._target.state.pos.clone(),
            }
            torch.save(data, self.destination_file)
        
        for key in self.reward_stats:
            self.reward_stats[key] = 0

    def reward(self, agent: Agent):
        is_first = self.world.policy_agents.index(agent) == 0

        if is_first:
            self.t += 1

            # count collisions
            for a in self.world.policy_agents:
                a.collision_rew[:] = 0

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if j <= i:
                        continue
                    collision = (self.world.get_distance(a, b) <= self.min_collision_distance)
                    if a.action_script is None:
                        a.collision_rew[collision] += self.collision_reward
                    if b.action_script is None:
                        b.collision_rew[collision] += self.collision_reward
            
            current_state = torch.stack([torch.cat((a.state.pos, a.state.vel,), dim=-1,) for a in self.world.policy_agents], dim=1,)

            self.phi_reward.zero_()

            for env in range(self.world.batch_dim):
                self.state_history[env].append(current_state[env].detach().clone())
                if len(self.state_history[env]) < self.phi_window:
                    continue
                history = torch.stack(list(self.state_history[env]), dim=0,) # (window, n_agents, features)

                if self.localise_phi: # phi local
                    positions = current_state[env, :, :2]
                    for i in range(len(self.world.policy_agents)):
                        distances = torch.linalg.vector_norm(positions[i] - positions, dim=-1,)
                        neighbours = torch.where(distances < self.communication_range_phi)[0]
                        if i not in neighbours:
                            neighbours = torch.cat((torch.tensor([i], device=positions.device,), neighbours,))
                        local_history = history[:, neighbours, :]
                        phi = self.phi(local_history)
                        delta_phi = (phi - self.phi_prev[env, i])
                        self.phi_prev[env, i] = phi
                        self.phi_reward[env, i] = ((phi if self.use_phi_instead_of_deltaphi else delta_phi) * self.phi_reward_scale)
                else: # phi global
                    phi = self.phi(history)
                    delta_phi = phi - self.phi_prev[env]
                    self.phi_prev[env] = phi
                    self.phi_reward[env] = ((phi if self.use_phi_instead_of_deltaphi else delta_phi) * self.phi_reward_scale)

        # separation reward

        distances = torch.stack([torch.linalg.vector_norm(agent.state.pos - a.state.pos, dim=-1) for a in self.world.agents if a != agent],dim=1,)
        squared = (distances - self.desired_distance).pow(2)
        mask = distances < self.communication_range_dist
        agents_dist_shaping = ((squared * mask).sum(-1) / (mask.sum(-1).clamp(min=1))) * self.dist_shaping_factor
        # agents_dist_shaping = squared.mean(-1) * self.dist_shaping_factor
            
        agent.dist_rew = agent.distance_shaping - agents_dist_shaping
        agent.distance_shaping = agents_dist_shaping

        collision = agent.collision_rew
        distance = agent.dist_rew
        phi = self.phi_reward[:, self.world.policy_agents.index(agent)] if self.localise_phi else self.phi_reward

        total = collision + distance + phi

        self.reward_stats["collision"] += collision.mean()
        self.reward_stats["distance"] += distance.mean()
        self.reward_stats["phi"] += phi.mean()

        return total
    
    def observation(self, agent: Agent):
        obs = torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self._target.state.pos,
                agent.sensors[0].measure(),
            ],
            dim=-1,
        )
        return obs

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "position": agent.state.pos.clone(),
            "speed": agent.state.vel.clone(),
            "collision_counter": agent.collision_rew.clone() / self.collision_reward,
            "t_position": self._target.state.pos.clone(),
            "t_speed": self._target.state.vel.clone()
        }

        return info
        
    def post_step(self):
        if self.t.mean() == 500 - 1:
            print(f'accumulated coll, dist, phi = {self.reward_stats["collision"]:.3f}, {self.reward_stats["distance"]:.3f}, {self.reward_stats["phi"]:.3f}')
        if self.boundary_type == "periodic":
            L_x = self.world_limit_x_dim
            L_y = self.world_limit_y_dim
            for agent in self.world.policy_agents:
                agent.state.pos[..., 0] = ((agent.state.pos[..., 0] + L_x) % (2 * L_x) - L_x)
                agent.state.pos[..., 1] = ((agent.state.pos[..., 1] + L_y) % (2 * L_y) - L_y)
        elif self.boundary_type == "clip":
            L_x = self.world_limit_x_dim
            L_y = self.world_limit_y_dim
            for agent in self.world.policy_agents:
                agent.state.pos[..., 0].clamp_(-L_x, L_x,)
                agent.state.pos[..., 1].clamp_(-L_y, L_y,)
        
    def gaus_entr(self, x):
        x = x - x.mean(dim=0)
        cov = (x.T @ x / (x.shape[0] - 1))
        cov += (1e-6 * torch.eye(cov.shape[0], device=x.device,))
        _, logdet = torch.slogdet(cov)
        return 0.5 * logdet
    
    def phi(self, states):
        t, n, features = states.shape
        joint_ent = self.gaus_entr(states.reshape(t, n * features))
        indiv_ent = torch.zeros((), device=states.device, dtype=states.dtype,)
        for i in range(n):
            indiv_ent += self.gaus_entr(states[:, i, :])
        return indiv_ent - joint_ent


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        # First calculate the closest point to a circle of radius circle_radius given the current position
        circle_origin = torch.zeros(1, 2)
        circle_radius = 0.3
        current_pos = observation[:, :2]
        v = current_pos - circle_origin
        closest_point_on_circ = (
            circle_origin + v / torch.linalg.norm(v, dim=1).unsqueeze(1) * circle_radius
        )

        # calculate the normal vector of the vector from the origin of the circle to that closest point
        # on the circle. Adding this scaled normal vector to the other vector gives us a target point we
        # try to reach, thus resulting in a circular motion.
        closest_point_on_circ_normal = torch.stack(
            [closest_point_on_circ[:, Y], -closest_point_on_circ[:, X]], dim=1
        )
        closest_point_on_circ_normal /= torch.linalg.norm(
            closest_point_on_circ_normal, dim=1
        ).unsqueeze(1)
        closest_point_on_circ_normal *= 0.1
        des_pos = closest_point_on_circ + closest_point_on_circ_normal

        # Move away from other agents and obstcles within visibility range
        lidar = observation[:, 6:18]
        object_visible = torch.any(lidar < 0.1, dim=1)
        _, object_dir_index = torch.min(lidar, dim=1)
        object_dir = object_dir_index / lidar.shape[1] * 2 * torch.pi
        object_vec = torch.stack([torch.cos(object_dir), torch.sin(object_dir)], dim=1)
        des_pos_object = current_pos - object_vec * 0.1
        des_pos[object_visible] = des_pos_object[object_visible]

        action = torch.clamp(
            (des_pos - current_pos) * 10,
            min=-u_range,
            max=u_range,
        )

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
