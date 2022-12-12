import gym
import gym_base
from q_learning import q_learning, policy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


grid_env = gym.make('gym_base/GridWorld-v0')
grid_env.reset()  # reset environment to a new, random state

print("Starting Q-learning")
Q, episode_avg_reward, episode_num_its = q_learning(grid_env)
print("Finding Policy")
p = policy(Q, grid_env.size)

# # plot the average reward per episode
plt.figure(figsize=(25, 20), dpi=60)
plt.plot(episode_avg_reward, alpha=0.5)
plt.plot(savgol_filter(episode_avg_reward, 1001, 3), linewidth=3)
plt.xlabel('Episode', fontsize=30)
plt.ylabel('Average Reward', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Average Reward per Episode', fontsize=40)
# plot the number of iterations per episode
plt.figure(figsize=(25, 20), dpi=60)
plt.plot(episode_num_its, alpha=0.5)
plt.plot(savgol_filter(episode_num_its, 1001, 3), linewidth=3)
plt.xlabel('Episode', fontsize=30)
plt.ylabel('Number of Iterations', fontsize=30)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.title('Number of Iterations per Episode', fontsize=40)

print("Displaying Policy")
plt.show(block=False)
grid_env.animate_scenario(p)
