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

x_min, y_min = grid_env.lower_xy_bounds
x_max, y_max = grid_env.upper_xy_bounds

goal_xy = grid_env.get_target_location()[:2]
goal_x = np.random.uniform(goal_xy[0] - grid_env.goal_size, goal_xy[0] + grid_env.goal_size)
goal_y = np.random.uniform(goal_xy[1] - grid_env.goal_size, goal_xy[1] + grid_env.goal_size)


initial_position = grid_env.sim_scenario.target_object.get_sim_pose(euler=True).position.tolist()
# init_p_index = grid_env.state_to_index(initial_position)

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

# print where Q is non zero
# import pdb; pdb.set_trace()
# for 


fig, ax = plt.subplots()
for y in range(grid_env.num_blocks_y):
    for x in range(grid_env.num_blocks_x):
        policy_pos = json.loads(p[x][y])['pos']
        policy_index = grid_env.state_to_index(policy_pos)
        reward = np.max(Q[x, y, :])

        loc_index = get_index_from_loc(x, y, grid_env)
        if grid_env.is_in_goal_region(loc_index):
            plot_cell(y, x, color='green')
        #     min_x, max_x = y, y+1
        #     min_y, max_y = x, x+1
        #     plt.plot([min_x, min_x], [min_y, max_y], color='green', linewidth=3)
        #     plt.plot([min_x, max_x], [max_y, max_y], color='green', linewidth=3)
        #     plt.plot([max_x, max_x], [max_y, min_y], color='green', linewidth=3)
        #     plt.plot([max_x, min_x], [min_y, min_y], color='green', linewidth=3)
        
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


pdb.set_trace()

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

