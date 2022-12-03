import gym
import gym_base
from q_learning import q_learning, policy
import matplotlib.pyplot as plt
import numpy as np


grid_env = gym.make('gym_base/GridWorld-v0')

# action = {'mode': 1, 'pos': (5, 0)}
grid_env.reset()
# observation, reward, terminated, _, _ = grid_env.step(action=action)
# print(observation)

# for row in range(0, grid_env.size):
#     for col in range(0, grid_env.size):
#         print("Row: %d, Col: %d" % (row, col))
#         print(grid_env.mode_handler.pos_is_in_range_for_grasp(np.array([row, col]), np.array([3, 3])))


print("Starting Q-learning")
Q, episode_avg_reward, episode_num_its = q_learning(grid_env)
print("Finding Policy")
p = policy(Q, grid_env.size)
for row in range(0, grid_env.size):
    for col in range(0, grid_env.size):
        print("row: %d, col: %d" % (row, col))
        print(Q[col, row])
    print()
grid_env.reset()
action = {'mode': 0, 'pos': (3, 1)}
observation, reward, terminated, _, _ = grid_env.step(action=action)
print('Reward after step 1: %f' % reward)
print('Observation after step 1: %s' % observation)
action = {'mode': 0, 'pos': (3, 3)}
observation, reward, terminated, _, _ = grid_env.step(action=action)
print('Reward after step 2: %f' % reward)
print('Observation after step 2: %s' % observation)

# for row in range(0, 2):
#     for col in range(0, 2):
#         print(f'state: ({row}, {col}), p: {p[row, col]}')

plt.figure()
plt.plot(episode_avg_reward)
plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.figure()
plt.plot(episode_num_its)
plt.xlabel('Episode')
plt.ylabel('Number of Iterations')
print("Displaying Policy")
grid_env.display_scenario(Q=Q, policy=p)

# grid_env.reset()

# action = {"mode": 2, "pos": (0, 0)}

# observation, reward, terminated, _, _ = grid_env.step(action=action)
# grid_env.display_scenario()
