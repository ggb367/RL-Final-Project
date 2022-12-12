import gym
import gym_base
from q_learning import q_learning, policy
import matplotlib.pyplot as plt
import numpy as np
import json

grid_env = gym.make('gym_base/GridWorld-v0')
grid_env.reset() 

x_min, y_min = grid_env.lower_xy_bounds
x_max, y_max = grid_env.upper_xy_bounds

goal_xy = grid_env.get_target_location()[:2]
goal_x = np.random.uniform(goal_xy[0] - grid_env.goal_size, goal_xy[0] + grid_env.goal_size)
goal_y = np.random.uniform(goal_xy[1] - grid_env.goal_size, goal_xy[1] + grid_env.goal_size)


initial_position = grid_env.sim_scenario.target_object.get_sim_pose(euler=True).position.tolist()

Q, episode_avg_reward, episode_num_its = q_learning(grid_env)
p = policy(Q, grid_env)


def get_index_from_loc(x, y, grid_env):
    lxy, uxy = grid_env.lower_xy_bounds,grid_env.upper_xy_bounds
    loc_index = (x * grid_env.discretize + lxy[0], y * grid_env.discretize - (uxy[1]- lxy[1])/2)
    return loc_index

def plot_cell(x, y, color):
    min_x, max_x = x, x+1
    min_y, max_y = y, y+1
    plt.plot([min_x, min_x], [min_y, max_y], color=color, linewidth=3)
    plt.plot([min_x, max_x], [max_y, max_y], color=color, linewidth=3)
    plt.plot([max_x, max_x], [max_y, min_y], color=color, linewidth=3)
    plt.plot([max_x, min_x], [min_y, min_y], color=color, linewidth=3)

fig, ax = plt.subplots()
for y in range(grid_env.num_blocks_y):
    for x in range(grid_env.num_blocks_x):
        policy_pos = json.loads(p[x][y])['pos']
        policy_index = grid_env.state_to_index(policy_pos)
        reward = np.max(Q[x, y, :])

        loc_index = get_index_from_loc(x, y, grid_env)
        if grid_env.is_in_goal_region(loc_index):
            plot_cell(y, x, color='green')
        
        plt.text(y + 0.3, x + 0.3, policy_index, color='black')
        plt.text(y + 0.3, x + 0.5, reward, color='red')

print("initial_position: ", initial_position)
init_index = get_index_from_loc(initial_position[0], initial_position[1], grid_env)
print("init_index", init_index)
plot_cell(init_index[1], init_index[0], color='blue')

xaxis = np.arange(0, grid_env.num_blocks_y + 1, 1)
yaxis = np.arange(0, grid_env.num_blocks_x + 1, 1)
plt.xticks(xaxis)
plt.yticks(yaxis)
plt.xlim(0, grid_env.num_blocks_y)
plt.ylim(0, grid_env.num_blocks_x)
plt.grid(True)
plt.show(block=False)
