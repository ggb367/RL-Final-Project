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

from gym_base.envs.grid_world.modes import ModeHandler
from gym_base.envs.grid_world.scenarios import ScenarioHandler

from gym_base.envs.grid_world.sim import get_contour_point, get_pose, get_ee_vel


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def setup_scenario(self):
        scenario_id = 1
        sim = True
        enable_realtime = True
        enable_gui = True
        self.sim_scenario = MultimodalEnv(scenario_id=scenario_id, sim=sim,
                                          enable_realtime=enable_realtime, enable_gui=enable_gui)

    def __init__(self, render_mode=None):
        self.setup_scenario()

        self.scenario_num = 1
        # TODO: convert to pybullet scenario handler
        self.scenario = ScenarioHandler(scenario=self.scenario_num)

        self.size = self.scenario.grid_size
        self._robot_arm_location = None
        self._object_location = None
        self._target_location = None
        self._obstacles_location = None
        self._prev_object_location = None
        self._object_graspable = None
        self.window_size = 512*2
        self.observation_space = spaces.Dict({
            "robot_arm":
            spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "object":
            spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "target":
            spaces.Box(0, self.size - 1, shape=(2, ), dtype=int),
            "obstacles":
            spaces.Sequence(spaces.Box(
                0, self.size - 1, shape=(2, ), dtype=int)),
            "object_graspable":
            spaces.MultiBinary(1),
        })

        # sim_size = 0
        ikea_size = np.array(self.sim_scenario.table_ikea.get_size())[0]
        # sim_size =

        self.observation_space_sim = spaces.Dict({
            "robot_arm":
            spaces.Box(-ikea_size, ikea_size, shape=(2, ), dtype=int),
            "object":
            spaces.Box(-ikea_size, ikea_size, shape=(2, ), dtype=int),
            "target":
            spaces.Box(-ikea_size, ikea_size, shape=(2, ), dtype=int),
            "obstacles":
            spaces.Sequence(
                spaces.Box(-ikea_size, ikea_size, shape=(2, ), dtype=int)),
            "object_graspable":
            spaces.MultiBinary(1),
        })
        self.action_space = spaces.Dict({"mode": spaces.Discrete(3),
                                         "pos": spaces.Tuple((spaces.Discrete(self.size),
                                                              spaces.Discrete(self.size)))})
        self.mode_handler = ModeHandler(
            grid_size=self.size, scenario_num=self.scenario_num)
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
            location = np.array(
                [random.randint(0, self.size - 1), random.randint(0, self.size - 1)])
            is_obs = False
            for obs in self._obstacles_location:
                if np.array_equal(location, obs):
                    is_obs = True
            if not is_obs:
                return location

    def reset_sim(self):
        self._robot_arm_pb_sim = self.sim_scenario.robot
        self._object_pb_sim = self.sim_scenario.target_object
        self._object_graspable_sim = True  # Add a graspable flag to the sim scenario
        self._target_pb_sim = self.sim_scenario.goal_node
        self._obstacles_pb_sim = self.sim_scenario.all_obstacles

    def reset(self, seed=None, random_start=False, options=None):
        super().reset(seed=seed)
        self.reset_sim()

        self._robot_arm_location = self.scenario.robot_arm_location

        if not random_start:
            self._object_location = self.scenario.object_location
        else:
            self._object_location = self.get_valid_random_location()

        self._object_graspable = self.scenario.object_graspable
        self._target_location = self.scenario.target_location
        self._obstacles_location = self.scenario.obstacles_location

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
        dest = [0.7, 0.2]
        end_ee_pose = get_pose(self.sim_scenario.target_object, dest)
        end_ee_pose.orientation = self.sim_scenario.robot.get_ee_pose().orientation

        countour_point = get_contour_point(
            end_ee_pose, self.sim_scenario.target_object)

        start_ee_pose = get_pose(self.sim_scenario.target_object, np.array(
            [countour_point.x, countour_point.y]))
        start_ee_pose.orientation = self.sim_scenario.robot.get_ee_pose().orientation
        start_ee_pose.position.z += 0.03

        end_ee_pose.position.z += 0.03

        # plt.plot([end_ee_pose.position.x], [end_ee_pose.position.y], color='blue')
        # plt.plot([start_ee_pose.position.x], [start_ee_pose.position.y], color='red')
        # plt.plot([countour_point.x], [countour_point.y], color='green')
        # plt.show()
        # start_ee_pose = self.sim_scenario.robot.get_ee_pose()
        # start_ee_pose.position.x = countour_point.x
        # start_ee_pose.position.y = countour_point.y
        # start_ee_pose.position.z = end_ee_pose.position.z

        init_joint_angles = self.sim_scenario.robot.get_ik_solution(
            start_ee_pose)
        final_joint_angles = self.sim_scenario.robot.get_ik_solution(
            end_ee_pose)

        init_joint_angles = [1.21146268,  1.01414402, -0.32507411, -2.25405194,  1.85247139,  2.81033492,
                             -0.07932621]
        final_joint_angles = [0.24571083,  1.46706513,  0.05270085, -0.93101708, -0.0385409,   2.3629429,
                              1.03453555]

        # pdb.set_trace()
        print("init: ", init_joint_angles)
        print("final:", final_joint_angles)

        # init_joint_angles = [-0.01436703, 1.72847498,  0.6815732, -
        #                      0.68954973, -0.85966426, 2.19816014, 1.24536637]

        # final_joint_angles = [0.01436703, 1.72847498,  0.6815732, -
        #                       0.68954973, -0.85966426, 2.19816014, 1.24536637]

        self.sim_scenario.robot.move_to_joint_angles(init_joint_angles, using_planner=False)
        # self.sim_scenario.robot.move_to_default_pose(using_planner=False)
        # self.sim_scenario.robot.move_to_joint_angles(final_joint_angles, using_planner=False)

        ee_vel_vec = get_ee_vel(self._object_pb_sim.get_sim_pose(
            euler=False), end_ee_pose, 1)

        # object_id = self._object_pb_sim.id
        # obstacle_ids = [obs.id for obs in self._obstacles_pb_sim]
        # ee_position_threshold = 1e-2
        # time_limit = 1

        target_ee_velocity = ee_vel_vec
        time_duration = 3
        object_id = self.sim_scenario.target_object.id
        # (ee_vel, duration, mode, object_id)
        self.sim_scenario.robot.execute_constant_ee_velocity(target_ee_velocity, time_duration, 'poke', object_id)

    # def apply_ee_velocity(self, end_ee_pose, target_ee_velocity, time_duration, ee_position_threshold=1e-2,
    #                       time_limit=1, skill_name='poke', collect_data=False, linear=False, **kwargs):

        # get_move_trajectort()
        # self.sim_scenario.robot.apply_impulse(end_ee_pose,
        #                                       ee_vel_vec)
        # self.sim_scenario.target_object.move_pose()

    def step(self, action):
        mode = self._action_mode[action["mode"]]
        dest = np.array(action["pos"])
        if mode == self.mode_handler.Mode.PUSH:
            self.apply_poke_in_sim(dest)

        self._prev_object_location = self._object_location.copy()
        self._object_location = self.mode_handler.move(
            self._object_location, mode, dest)
        reward = self.calc_reward()
        terminated = np.array_equal(
            self._object_location, self._target_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def a_star_distance(self, start, goal):
        """
        A* search to find the shortest path from start to goal.
        """
        # define grid world in networkx graph
        G = nx.grid_2d_graph(self.size, self.size)
        tuple_list = [tuple(item) for item in self._obstacles_location]
        for row in range(self.size):
            for col in range(self.size):
                if any((row, col) == item for item in tuple_list):
                    G.remove_node((row, col))
                if row < self.size - 1:
                    G.add_node((row, col))
                    G.add_node((row + 1, col))
                    G.add_edge((row, col), (row + 1, col))
                if col < self.size - 1:
                    G.add_node((row, col))
                    G.add_node((row, col + 1))
                    G.add_edge((row, col), (row, col + 1))

        # find the shortest path
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
        pix_square_size = (self.window_size / self.size
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
