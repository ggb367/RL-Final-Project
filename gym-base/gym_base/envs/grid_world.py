from argparse import Action
import gym
from gym import spaces
import pygame
import numpy as np
import random


class GridWorldEnv(gym.Env):
    class Action:
        R = np.array([1, 0])
        U = np.array([0, 1])
        L = np.array([-1, 0])
        D = np.array([0, -1])

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Dict({
            "agent":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
            "target":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
            "obstacle":
            spaces.Box(0, size - 1, shape=(2, ), dtype=int),
        })
        self.action_space = spaces.Discrete(4)  # TODO
        self._action_to_direction = {
            0: self.Action.R,
            1: self.Action.U,
            2: self.Action.L,
            3: self.Action.D,
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location,
            "obstacle": self._obstacle_location,
        }

    def _get_info(self):
        return {
            "distance":
            np.linalg.norm(self._agent_location - self._target_location, ord=1)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = np.array([1, 0])
        self._target_location = np.array([3, 3])
        self._obstacle_location = np.array([1, 2])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = self.get_next_location(direction)

        reward = self.calc_reward()  # TODO
        terminated = np.array_equal(
            self._agent_location, self._target_location)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def get_next_location(self, direction):
        return np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

    def calc_reward(self):  # TODO
        return 1

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
            (self._agent_location + 0.5) * pix_square_size,
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
