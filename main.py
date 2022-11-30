import gym
import gym_base

grid_env = gym.make('gym_base/GridWorld-v0')

grid_env.reset()
action = {"mode": 2, "pos": (3, 0)}

observation, reward, terminated, _, _ = grid_env.step(action=action)
grid_env.display_scenario()
