from argparse import Action
import gym
from gym import spaces
import pygame
import numpy as np
import random

from gym_base.envs.grid_world.modes import ModeHandler


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self._robot_arm_location = np.array([0, 0])
        self._object_location = np.array([1, 0])
        self._target_location = np.array([3, 2])
        self._obstacles_location = [np.array([2, 2]),
                                    np.array([1, 1]),
                                    np.array([0, 1])]
        self._prev_object_location = self._object_location.copy()

        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Dict({
            "robot_arm":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
            "object":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
            "target":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
            "obstacles":
            spaces.Sequence(spaces.Box(0, size - 1, shape=(2, ), dtype=int)),
        })

        self.action_space = spaces.Dict({"mode": spaces.Discrete(3),
                                         "pos": spaces.Tuple((spaces.Discrete(4),
                                                              spaces.Discrete(4)))})
        self.mode_handler = ModeHandler(grid_size=size)

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
            "obstacles": self._obstacles_location
        }

    def _get_info(self):
        return {
            "distance":
            np.linalg.norm(self._object_location -
                           self._target_location, ord=1)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._robot_arm_location = np.array([0, 0])
        self._object_location = np.array([1, 0])
        self._target_location = np.array([3, 2])
        self._obstacles_location = [np.array([2, 2]),
                                    np.array([1, 1]),
                                    np.array([0, 1])]

        self.mode_handler.reset(self._robot_arm_location,
                                self._object_location,
                                self._target_location,
                                self._obstacles_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        mode = self._action_mode[action["mode"]]
        dest = np.array(action["pos"])

        self._prev_object_location = self._object_location.copy()

        self._object_location = self.mode_handler.move(
            start=self._object_location,
            mode=mode, dest=dest)

        reward = self.calc_reward()  # TODO
        terminated = np.array_equal(
            self._object_location, self._target_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def calc_reward(self):  # TODO
        """
        Calculate the reward for the current state.
        """
        # base reward on action taken and how close the object is to the goal after the action
        # 1 unit of distance from goal after action = -1 reward
        # 2 units of distance from goal after action = -2 reward etc...
        # -1000 if action gets object stuck under tunnel
        # -1000 if action gets object to fall off of table

        reward = 0

        # calc distance to goal
        object_distance_to_goal = np.linalg.norm(self._object_location - self._target_location, ord=1)
        reward -= object_distance_to_goal
        # calc distance traveled
        distance_traveled = np.linalg.norm(self._object_location - self._prev_object_location, ord=1)
        reward += distance_traveled
        # check to see if object is under tunnel
        if self.object_under_tunnel():  # TODO
            reward -= 1000  # TODO: making it -1000 might make it unstable, need to test it
        # check to see if object is off table
        if self.object_off_table():  # TODO
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
