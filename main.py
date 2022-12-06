import gym
import gym_base
from q_learning import q_learning, policy
import matplotlib.pyplot as plt


grid_env = gym.make('gym_base/GridWorld-v0')
grid_env.reset()  # reset environment to a new, random state

print("Starting Q-learning")
Q, episode_avg_reward, episode_num_its = q_learning(grid_env)
print("Finding Policy")
p = policy(Q, grid_env.size)
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
grid_env.animate_scenario(p)

# extra code:
# to create an action and test it:
# action = {'mode': 0, 'pos': (3, 1)} where mode is 0 for grasp, 1 for push, 2 for poke
# observation, reward, terminated, _, _ = grid_env.step(action=action)
# print('Reward: %f' % reward)
# print('Observation: %s' % observation)
