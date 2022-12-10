import time

import gym
from gym import spaces
import pygame
import numpy as np
import random
import networkx as nx
import pdb
import pybullet as pb
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
# import shapely affine
from shapely import affinity

from npm_base import Point, Quaternion, Pose, convert_orientation
from robot_sim_envs import MultimodalEnv
from multimodal_planning_v2 import PathPlanner

from gym_base.envs.grid_world.modes import SimModeHandler
from gym_base.envs.grid_world.scenarios import ScenarioHandler

from gym_base.envs.grid_world.sim import get_contour_point, get_pose, get_ee_vel


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def setup_scenario(self):
        scenario_id = 1
        sim = True
        enable_gui = True
        self.enable_realtime = True
        return MultimodalEnv(scenario_id=scenario_id, sim=sim,
                             enable_realtime=self.enable_realtime, enable_gui=enable_gui)

    def __init__(self, render_mode=None):
        self.sim_scenario = self.setup_scenario()

        self.discretize = 0.0635

        self.ikea_size = np.array(self.sim_scenario.table_ikea.get_size())[
            :2]  # only want x and y

        offset = self.sim_scenario.table_ikea.node.x - self.ikea_size[0]/2
        self.lower_xy_bounds = np.array(
            [offset, -self.ikea_size[1] / 2])  # x min, y min
        self.upper_xy_bounds = np.array(
            [self.ikea_size[0] + offset, self.ikea_size[1] / 2])  # x max, y max

        self.ikea_z = self.sim_scenario.target_object.get_sim_pose(
            euler=True).position.z

        self.num_blocks_x = int(self.ikea_size[0] / self.discretize)
        self.num_blocks_y = int(self.ikea_size[1] / self.discretize)

        # self._robot_arm_location = None
        # self._object_location = None
        # self._target_location = None
        # self._obstacles_location = None
        # self._prev_object_location = None
        # self._object_graspable = None
        # self._obstacles_list = None

        self.observation_space = spaces.Dict(
            {
                "object":
                spaces.Box(low=self.lower_xy_bounds,
                           high=self.upper_xy_bounds, shape=(2,), dtype=float),
                "target":
                spaces.Box(low=self.lower_xy_bounds,
                           high=self.upper_xy_bounds, shape=(2,), dtype=float),
                "obstacles":
                spaces.Sequence(
                    spaces.Box(low=self.lower_xy_bounds, high=self.upper_xy_bounds, shape=(2,), dtype=float)),
            })

        self.action_space = spaces.Dict(
            {"mode": spaces.Discrete(3),
             "pos": spaces.Box(low=self.lower_xy_bounds, high=self.upper_xy_bounds, shape=(2,), dtype=float)})

        self.mode_handler = SimModeHandler(self.discretize, self.enable_realtime, self.ikea_z)
        self._action_mode = {
            0: self.mode_handler.Mode.GRASP,
            1: self.mode_handler.Mode.PUSH,
            2: self.mode_handler.Mode.POKE,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "object": self._object_location,
            "target": self._target_location,
            "obstacles": self._obstacles_location,
        }

    def _get_info(self):
        return {
            "distance":
                np.linalg.norm(self._object_location -
                               self._target_location, ord=1)
        }


    def state_to_index(self, state):
        interval = self.discretize
        x = state[0]
        y = state[1]
        x_index = int(x / interval)
        y_index = int(y / interval)
        return [x_index, y_index]


    def get_valid_random_location(self):
        while True:
            location = np.array([random.uniform(self.lower_xy_bounds[0], self.upper_xy_bounds[0]),
                                 random.uniform(self.lower_xy_bounds[1], self.upper_xy_bounds[1])])
            location = np.round(location / self.discretize) * self.discretize
            is_obs = False
            for obs in self._obstacles_location:
                if np.array_equal(location, obs):
                    is_obs = True
            if not is_obs:
                return location

    def reset(self, seed=None, random_start=False, options=None):
        super().reset(seed=seed)
        self.sim_scenario.reset()

        if not random_start:
            self._object_location = self.sim_scenario.target_object.get_position()[
                :2]
        else:
            self._object_location = self.get_valid_random_location()

        self._target_location = [
            self.sim_scenario.goal_node.x, self.sim_scenario.goal_node.y]

        obstacles_list = self.sim_scenario.all_obstacles
        self._obstacles_location = []
        for obstacle in obstacles_list:
            obstacle_size = obstacle.get_size()
            obstacle_pos = obstacle.get_position()
            obstacle_grid_spots = []
            for i in range(int(obstacle_size[0] / self.discretize)):
                for j in range(int(obstacle_size[1] / self.discretize)):
                    obstacle_grid_spots.append([obstacle_pos[0] + i * self.discretize,
                                                obstacle_pos[1] + j * self.discretize])
            self._obstacles_location.extend(obstacle_grid_spots)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def position_on_table(self, pos):
        return self.lower_xy_bounds[0] <= pos[0] <= self.upper_xy_bounds[0] and \
            self.lower_xy_bounds[1] <= pos[1] <= self.upper_xy_bounds[1]

    def step(self, action):
        mode = self._action_mode[action["mode"]]
        dest = np.array(action["pos"])

        self._prev_object_location = self._object_location.copy()

        self._object_location = self.mode_handler.move(mode, dest, self.sim_scenario.target_object, self.sim_scenario.robot)

        reward = self.calc_reward()
        object_is_on_table = self.position_on_table(self._object_location)

        index = self.state_to_index(self._object_location)
        index_in_range = index[0] < self.num_blocks_x and index[1] < self.num_blocks_y

        terminated = np.array_equal( self._object_location, self._target_location) and object_is_on_table and not index_in_range

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def within_radius(self, node, tuple_list, radius):
        for tuple in tuple_list:
            if np.linalg.norm(np.array(node) - np.array(tuple)) < radius:
                return True
        return False

    def a_star_distance(self, start, goal):
        x_nodes = np.arange(
            self.lower_xy_bounds[0],
            self.upper_xy_bounds[0],
            self.discretize)
        y_nodes = np.arange(
            self.lower_xy_bounds[1],
            self.upper_xy_bounds[1],
            self.discretize)
        G = nx.grid_2d_graph(
            x_nodes, y_nodes)

        obs_list = [tuple(item) for item in self._obstacles_location]

        nodes = list(G.nodes)
        for node in nodes:
            if self.within_radius(node, obs_list, self.discretize):
                G.remove_node(node) # TODO: this shouldn't have the if, right?
                for neighbor in G.neighbors(node):
                    if neighbor in obs_list:
                        G.remove_edge(node, neighbor)

        start_node = min(G.nodes(), key=lambda x: np.linalg.norm(
            np.array(x) - np.array(start)))
        G.add_node(tuple(start))
        G.add_edge(tuple(start), start_node)

        goal_node = min(G.nodes(), key=lambda x: np.linalg.norm(
            np.array(x) - np.array(goal)))

        G.add_edge(tuple(goal), goal_node)
        G.add_node(tuple(goal))

        path = nx.astar_path(G, tuple(start), tuple(goal))

        return len(path)

    def calc_reward(self):
        reward = 0

        object_distance_to_goal = self.a_star_distance(self._object_location, self._target_location)
        reward -= object_distance_to_goal

        if np.array_equal(self._object_location, self._target_location):
            reward += 500
        else:
            reward -= 500

        # if object doesn't move, penalty
        if np.array_equal(self._object_location, self._prev_object_location):
            reward -= 1000
        return reward
