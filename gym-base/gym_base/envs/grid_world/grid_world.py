import gym
from gym import spaces
import pygame
import numpy as np
import random
import networkx as nx

from gym_base.envs.grid_world.modes import ModeHandler
from gym_base.envs.grid_world.scenarios import ScenarioHandler


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        self.scenario_num = 1
        self.scenario = ScenarioHandler(scenario=self.scenario_num)  # TODO: convert to pybullet scenario handler
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
            spaces.Sequence(spaces.Box(0, self.size - 1, shape=(2, ), dtype=int)),
            "object_graspable":
            spaces.MultiBinary(1),
        })
        self.action_space = spaces.Dict({"mode": spaces.Discrete(3),
                                         "pos": spaces.Tuple((spaces.Discrete(self.size),
                                                              spaces.Discrete(self.size)))})
        self.mode_handler = ModeHandler(grid_size=self.size, scenario_num=self.scenario_num)
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
            location = np.array([random.randint(0, self.size - 1), random.randint(0, self.size - 1)])
            is_obs = False
            for obs in self._obstacles_location:
                if np.array_equal(location, obs):
                    is_obs = True
            if not is_obs:
                return location

    def reset(self, seed=None, random_start = False, options=None):
        super().reset(seed=seed)

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

    def step(self, action):
        mode = self._action_mode[action["mode"]]
        dest = np.array(action["pos"])

        self._prev_object_location = self._object_location.copy()

        self._object_location = None # TODO update this to call pybullet
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
        object_distance_to_goal = self.a_star_distance(self._object_location, self._target_location)
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
