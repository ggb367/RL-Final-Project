import gym
import gym_base

grid_env = gym.make('gym_base/GridWorld-v0')

grid_env.reset()
grid_env.step(1)

