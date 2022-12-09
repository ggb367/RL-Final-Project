import time

import gym
from gym import spaces
import pygame
import numpy as np
import random
import networkx as nx
import pdb

import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
# import shapely affine
from shapely import affinity

from npm_base import Point, Quaternion, Pose, convert_orientation
from robot_sim_envs import MultimodalEnv
from multimodal_planning_v2 import PathPlanner

from gym_base.envs.grid_world.modes import ModeHandler, SimModeHandler
from gym_base.envs.grid_world.scenarios import ScenarioHandler

from gym_base.envs.grid_world.sim import get_contour_point, get_pose, get_ee_vel


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def setup_scenario(self):
        scenario_id = 1
        sim = True
        enable_gui = True
        print("Setting up scenario {}".format(scenario_id))
        # fig, ax = plt.subplots()
        self.sim_scenario = MultimodalEnv(scenario_id=scenario_id, sim=sim,
                                          enable_realtime=self.enable_realtime, enable_gui=enable_gui)
        # plt.show()
        print("Done setting up scenario {}".format(scenario_id))

    def __init__(self, render_mode=None):
        self.sim_scenario = None
        self.enable_realtime = True
        self.setup_scenario()

        self.scenario_num = 1
        self.discritize = 0.0635  # size of the block in meters
        # TODO: convert to pybullet scenario handler
        self.scenario = ScenarioHandler(scenario=self.scenario_num)

        self.shape = np.array(self.sim_scenario.table_ikea.get_size())[:2]  # only want x and y
        self.lower_xy_bounds = np.array([0, -self.shape[1] / 2])
        self.upper_xy_bounds = np.array([self.shape[0], self.shape[1] / 2])

        self.size = self.scenario.grid_size
        self.num_blocks_x = int(self.shape[0] / self.discritize)
        self.num_blocks_y = int(self.shape[1] / self.discritize)
        self._robot_arm_location = None
        self._object_location = None
        self._target_location = None
        self._obstacles_location = None
        self._prev_object_location = None
        self._object_graspable = None
        self._obstacles_list = None
        self.window_size = 512 * 2

        self.observation_space = spaces.Dict({
            "robot_arm":
                spaces.Box(low=self.lower_xy_bounds, high=self.upper_xy_bounds, shape=(2,), dtype=float),
            "object":
                spaces.Box(low=self.lower_xy_bounds, high=self.upper_xy_bounds, shape=(2,), dtype=float),
            "target":
                spaces.Box(low=self.lower_xy_bounds, high=self.upper_xy_bounds, shape=(2,), dtype=float),
            "obstacles":
                spaces.Sequence(
                    spaces.Box(low=self.lower_xy_bounds, high=self.upper_xy_bounds, shape=(2,), dtype=float)),
            "object_graspable":
                spaces.MultiBinary(1),
        })

        # self.action_space = spaces.Dict({"mode": spaces.Discrete(3),
        #                                  "pos": spaces.Tuple((spaces.Discrete(self.num_blocks_x),
        #                                                       spaces.Discrete(self.num_blocks_y)))})
        self.action_space = spaces.Dict({"mode": spaces.Discrete(3),
                                         "pos": spaces.Box(low=self.lower_xy_bounds, high=self.upper_xy_bounds,
                                                           shape=(2,), dtype=float)})  # continuous action space
        self.mode_handler = SimModeHandler(self.discritize, self.enable_realtime)
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
            "robot_arm": self._robot_arm_location,
            "object": self._object_location,
            "target": self._target_location,
            "obstacles": self._obstacles_location,
            "object_graspable": self._object_graspable
        }

    def _get_info(self):
        return {
            "distance":
                np.linalg.norm(self._object_location -
                               self._target_location, ord=1)
        }

    def get_valid_random_location(self):
        """
        Get a random location that is not an obstacle.
        """
        while True:
            location = np.array([random.uniform(self.lower_xy_bounds[0], self.upper_xy_bounds[0]),
                                 random.uniform(self.lower_xy_bounds[1], self.upper_xy_bounds[1])])
            # round location to multiple of self.discritize
            location = np.round(location / self.discritize) * self.discritize
            is_obs = False
            for obs in self._obstacles_location:
                if np.array_equal(location, obs):
                    is_obs = True
            if not is_obs:
                return location

    def reset(self, seed=None, random_start=False, options=None):
        super().reset(seed=seed)
        self.sim_scenario.reset()

        self._robot_arm_location = self.scenario.robot_arm_location  # TODO: Delete this, not used anywhere

        if not random_start:
            self._object_location = self.sim_scenario.target_object.get_position()[:2]
        else:
            self._object_location = self.get_valid_random_location()

        self._object_graspable = self.scenario.object_graspable

        self._target_location = [self.sim_scenario.goal_node.x, self.sim_scenario.goal_node.y]
        # list of scene objects
        self._obstacles_list = self.sim_scenario.all_obstacles
        # list of obstacle locations in the grid, if grid spot is occupied by an obstacle include it in the list
        self._obstacles_location = []
        for obstacle in self._obstacles_list:
            obstacle_size = obstacle.get_size()
            obstacle_pos = obstacle.get_position()
            # grid spots occupied by the obstacle
            obstacle_grid_spots = []
            for i in range(int(obstacle_size[0] / self.discritize)):
                for j in range(int(obstacle_size[1] / self.discritize)):
                    obstacle_grid_spots.append([obstacle_pos[0] + i * self.discritize,
                                                obstacle_pos[1] + j * self.discritize])
            self._obstacles_location.extend(obstacle_grid_spots)

        self.mode_handler.reset(self._robot_arm_location,
                                self._object_location,
                                self._target_location,
                                self._obstacles_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def display_scenario(self, policy=None, Q=None):
        self.scenario.display(policy=policy, Q=Q)

    def animate_scenario(self, policy):
        self.scenario.animate(policy=policy)

    def apply_poke_in_sim(self, dest):
        # dest = [0.7, 0.2]
        end_ee_pose = get_pose(self.sim_scenario.target_object, dest)
        end_ee_pose.orientation = self.sim_scenario.robot.get_ee_pose().orientation

        countour_point = get_contour_point(
            end_ee_pose, self.sim_scenario.target_object)

        start_ee_pose = get_pose(self.sim_scenario.target_object, np.array(
            [countour_point.x, countour_point.y]))
        start_ee_pose.orientation = self.sim_scenario.robot.get_ee_pose().orientation
        start_ee_pose.position.z += 0.03

        end_ee_pose.position.z += 0.03

        init_joint_angles = self.sim_scenario.robot.get_ik_solution(
            start_ee_pose)
        final_joint_angles = self.sim_scenario.robot.get_ik_solution(
            end_ee_pose)

        # init_joint_angles = [1.21146268,  1.01414402, -0.32507411, -2.25405194,  1.85247139,  2.81033492,
        #                      -0.07932621]
        # final_joint_angles = [0.24571083,  1.46706513,  0.05270085, -0.93101708, -0.0385409,   2.3629429,
        #                       1.03453555]

        print("init: ", init_joint_angles)
        print("final:", final_joint_angles)

        self.sim_scenario.robot.move_to_joint_angles(init_joint_angles, using_planner=False)

        ee_vel_vec = get_ee_vel(self.sim_scenario.target_object.get_sim_pose(
            euler=False), end_ee_pose, 1)

        target_ee_velocity = ee_vel_vec
        time_duration = 3
        object_id = self.sim_scenario.target_object.id
        self.sim_scenario.robot.execute_constant_ee_velocity(target_ee_velocity, time_duration, 'poke', object_id)

    def step(self, action):
        mode = self._action_mode[action["mode"]]
        dest = np.array(action["pos"])
        self._object_location = self.mode_handler.move(mode, dest, self.sim_scenario.target_object,
                                                       self.sim_scenario.robot)
        self._prev_object_location = self._object_location.copy()
        # self._object_location = self.mode_handler.move(
        #     self._object_location, mode, dest)
        reward = self.calc_reward()
        terminated = np.array_equal(
            self._object_location, self._target_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def within_radius(self, node, tuple_list, radius):
        for tuple in tuple_list:
            if np.linalg.norm(np.array(node) - np.array(tuple)) < radius:
                return True
        return False

    def a_star_distance(self, start, goal):
        """
        A* search to find the shortest path from start to goal.
        """
        # define an nx.graph_2d_grid discritized graph with nodes at each grid spot
        x_nodes = np.arange(self.lower_xy_bounds[0], self.upper_xy_bounds[0], self.discritize)
        y_nodes = np.arange(self.lower_xy_bounds[1], self.upper_xy_bounds[1], self.discritize)
        G = nx.grid_2d_graph(x_nodes, y_nodes)

        tuple_list = [tuple(item) for item in self._obstacles_location]

        # for loop to remove nodes and edges that are occupied by obstacles
        nodes = list(G.nodes)
        for node in nodes:
            if self.within_radius(node, tuple_list, self.discritize):
                G.remove_node(node)
            else:
                for neighbor in G.neighbors(node):
                    if neighbor in tuple_list:
                        G.remove_edge(node, neighbor)

        # for row in np.arange(self.lower_xy_bounds[0], self.upper_xy_bounds[0], self.discritize):
        #     for col in np.arange(self.lower_xy_bounds[1], self.upper_xy_bounds[1], self.discritize):
        #         if any((row, col) == item for item in tuple_list):
        #             G.remove_node((row, col))
        #         if row < self.shape[0] - 1:
        #             G.add_node((row, col))
        #             G.add_node((row + 1, col))
        #             G.add_edge((row, col), (row + 1, col))
        #         if col < self.shape[1] - 1:
        #             G.add_node((row, col))
        #             G.add_node((row, col + 1))
        #             G.add_edge((row, col), (row, col + 1))
        # add edge from start to closest node
        start_node = min(G.nodes(), key=lambda x: np.linalg.norm(np.array(x) - np.array(start)))
        G.add_node(tuple(start))
        G.add_edge(tuple(start), start_node)
        # add edge from goal to closest node
        goal_node = min(G.nodes(), key=lambda x: np.linalg.norm(
            np.array(x) - np.array(goal)))

        G.add_edge(tuple(goal), goal_node)
        G.add_node(tuple(goal))

        path = nx.astar_path(G, tuple(start), tuple(goal))
        # return path length
        return len(path)

    def calc_reward(self):
        """
        Calculate the reward for the current state.
        """

        reward = 0

        # calc distance to goal
        object_distance_to_goal = self.a_star_distance(
            self._object_location, self._target_location)
        reward -= object_distance_to_goal

        # reach goal or not
        if np.array_equal(self._object_location, self._target_location):
            reward += 500
        else:
            reward -= 500

        # if object doesn't move, penalty
        if np.array_equal(self._object_location, self._prev_object_location):
            reward -= 1000
        return reward

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (self.window_size / self.discritize
                           )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._object_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)),
                                axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
