import gym
import gym_base
from q_learning import q_learning, policy, test_plot
import matplotlib.pyplot as plt
import numpy as np
import json

import pdb


import pybullet as pb

grid_env = gym.make('gym_base/GridWorld-v0')
grid_env.reset()

Q, episode_avg_reward, episode_num_its = q_learning(grid_env)
p = policy(Q, grid_env)


# def get_index_from_loc(x, y, grid_env):
#     lxy, uxy = grid_env.lower_xy_bounds, grid_env.upper_xy_bounds
#     loc_index = ((x - lxy[0])/grid_env.discretize,
#                  (y + (uxy[1] - lxy[1])/2)/grid_env.discretize)
#     return loc_index

def get_location_from_index(x, y, grid_env):
    lxy, uxy = grid_env.lower_xy_bounds, grid_env.upper_xy_bounds
    loc = (x * grid_env.discretize + lxy[0],
           y * grid_env.discretize - (uxy[1] - lxy[1])/2)
    return loc


def plot_cell(x, y, color):
    min_x, max_x = x, x+1
    min_y, max_y = y, y+1
    plt.plot([min_x, min_x], [min_y, max_y], color=color, linewidth=3)
    plt.plot([min_x, max_x], [max_y, max_y], color=color, linewidth=3)
    plt.plot([max_x, max_x], [max_y, min_y], color=color, linewidth=3)
    plt.plot([max_x, min_x], [min_y, min_y], color=color, linewidth=3)


fig, ax = plt.subplots()
fig.set_size_inches(20, 12)

for y in range(grid_env.num_blocks_y):
    for x in range(grid_env.num_blocks_x):
        
        policy_pos = json.loads(p[x][y])['pos']
        mode = json.loads(p[x][y])['mode']

        policy_index = grid_env.state_to_index(policy_pos)
        reward = np.max(Q[x, y, :])

        index_location = get_location_from_index(x, y, grid_env)
        if grid_env.is_in_goal_region(index_location):
            plot_cell(y, x, color='green')

        plt.text(y + 0.3, x + 0.3, (policy_index[1], policy_index[0]), color='black')
        plt.text(y + 0.3, x + 0.7, mode, color='blue')
        plt.text(y + 0.3, x + 0.5, reward, color='red')

init_pos_x, init_pos_y = grid_env.initial_target_object_pose.position.x, grid_env.initial_target_object_pose.position.y
init_index = grid_env.state_to_index([init_pos_x, init_pos_y])
plot_cell(init_index[1], init_index[0], color='blue')

xaxis = np.arange(0, grid_env.num_blocks_y + 1, 1)
yaxis = np.arange(0, grid_env.num_blocks_x + 1, 1)
plt.xticks(xaxis)
plt.yticks(yaxis)
plt.xlim(0, grid_env.num_blocks_y)
plt.ylim(0, grid_env.num_blocks_x)
plt.grid(True)
plt.savefig(f'demo/{grid_env.timestamp}_{grid_env.scenario_id}.png', dpi=300)
plt.show()

# policy_init_pos = json.loads(p[init_p_index[0]][init_p_index[1]])
# target_pos_for_init_pos = policy_init_pos['pos']
# print("target_pos_for_init_pos: ", target_pos_for_init_pos)
# in_goal_reigon = grid_env.is_in_goal_region(target_pos_for_init_pos)
# print("in goal reigon: ", in_goal_reigon)

# position = [target_pos_for_init_pos[0], target_pos_for_init_pos[1], grid_env.ikea_z]
# position_ = [position[0]+0.01, position[1]+0.01, grid_env.ikea_z]

# pb.removeAllUserDebugItems()
# debug_id = pb.addUserDebugLine(position, position_, lineWidth=10, lineColorRGB=[0, 0, 255])
# grid_env.sim_scenario.display_gui()
