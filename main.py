import gym
import gym_base
from q_learning import q_learning, policy, test_plot
import matplotlib.pyplot as plt
import numpy as np

grid_env = gym.make('gym_base/GridWorld-v0')
grid_env.reset()  # reset environment to a new, random state

# action = {'mode': 2, 'pos': (0.3, 0.2)}


x_min, y_min = grid_env.lower_xy_bounds
x_max, y_max = grid_env.upper_xy_bounds

# # 0.22175 -0.8001 1.0218500000000001 0.8001

# x = np.random.uniform(x_min + 0.02 , x_max - 0.5)
# y = np.random.uniform(y_min + 0.02 , y_max - 0.02)

# z = grid_env.z
# point = [x, y, z]
# point = [0.49666240753590807, -0.5221709381505836, -0.0825599283363301]
# point2 = [x + 0.01, y + 0.01,z]

# print("point: ", point)

# line = [point, point2]
# grid_env.plot_pybullet(lines=[line], color=[255, 0, 0], linewidth=10)

# action = {'mode': 0, 'pos': point[:2]}
# observation, reward, terminated, _, _ = grid_env.step(action)

# grid_env.sim_scenario.display_gui()
# print(reward)


print("Starting Q-learning")
Q, episode_avg_reward, episode_num_its = q_learning(grid_env)
print("Finding Policy")
# p = policy(Q, grid_env.size)
# print(p)
# for row in range(0, grid_env.size):
#     for col in range(0, grid_env.size):
#         print("row: %d, col: %d" % (row, col))
#         print(Q[col, row])
#     print()

# # plot the average reward per episode
# plt.figure()
# plt.plot(episode_avg_reward)
# plt.xlabel('Episode')
# plt.ylabel('Average Reward')
# plt.title('Average Reward per Episode')
# # plot the number of iterations per episode
# plt.figure()
# plt.plot(episode_num_its)
# plt.xlabel('Episode')
# plt.ylabel('Number of Iterations')
# print("Displaying Policy")
# display the policy, passing Q and p print them on the graph
# grid_env.animate_scenario(p)

# extra code:
# to create an action and test it:
# action = {'mode': 0, 'pos': (3, 1)} where mode is 0 for grasp, 1 for push, 2 for poke
# observation, reward, terminated, _, _ = grid_env.step(action=action)
# print('Reward: %f' % reward)
# print('Observation: %s' % observation)
