import gym
import gym_base

grid_env = gym.make('gym_base/GridWorld-v0')

grid_env.reset()

action = {"mode": 0, "pos": (3, 3)}
observation, reward, terminated, _, _ = grid_env.step(action=action)
# print("observation: ", observation)
# print("agent_location: ", observation["agent"])