import pdb

import numpy as np
import random

from tqdm import tqdm
import pybullet as pb

def test_plot(env):
    lxy = env.lower_xy_bounds
    uxy = env.upper_xy_bounds
    z = env.ikea_z    
    N = 100
    for i in range(N):
        random_x = random.uniform(lxy[0], uxy[0])
        random_y = random.uniform(lxy[1], uxy[1])
        index = env.state_to_index([random_x, random_y])
        point1 = [random_x, random_y, z]
        point2 = [random_x + 0.01, random_y + 0.01, z]
        pb.addUserDebugLine(point1, point2, lineWidth=10, lineColorRGB=[0, 0, 255])


def q_learning(env):
    # test_plot(env)
    # return None, None, None

    learning_rate = 0.4
    epsilon = 0.5
    discount_factor = 0.95

    num_of_episodes = 50
    number_of_iterations = 40

    num_of_rows = env.num_blocks_x
    num_of_columns = env.num_blocks_y

    # There are 3 modes, which each can have (grid_size, grid_size) poses
    num_of_actions = 3 * env.num_blocks_x * env.num_blocks_y

    Q = np.zeros((num_of_rows, num_of_columns, num_of_actions))

    episode_avg_reward = np.zeros(num_of_episodes)
    episode_num_its = np.zeros(num_of_episodes)
    debug_id = None

    for ep in tqdm(range(num_of_episodes)):
        observation, _ = env.reset(random_start=False)
        current_xy = observation['object']
        current_state = env.state_to_index(current_xy)
        avg_reward = 0
        for it in range(number_of_iterations):
            if debug_id is not None:
                pb.removeUserDebugItem(debug_id)

            action = get_epsilon_greedy_action(Q, current_state, num_of_actions, epsilon, env)
            env_action = _env_action(action, env)

            position = [env_action['pos'][0], env_action['pos'][1], env.ikea_z]
            position_ = [position[0]+0.01, position[1]+0.01, env.ikea_z]
            debug_id = pb.addUserDebugLine(position, position_, lineWidth=10, lineColorRGB=[0, 0, 255])

            observation, reward, terminated, _, _ = env.step(env_action)
            avg_reward += reward

            if terminated:
                episode_avg_reward[ep] = avg_reward / (it + 1)
                episode_num_its[ep] = it
                break


            next_xy = observation['object']
            next_state = env.state_to_index(next_xy)


            Q[current_state[0], current_state[1], action] += learning_rate * (reward + discount_factor * np.max(
                Q[next_state[0], next_state[1], :]) - Q[current_state[0], current_state[1], action])

            current_state = next_state

            if it == number_of_iterations - 1:
                episode_avg_reward[ep] = avg_reward / (it + 1)
                episode_num_its[ep] = it
    return Q, episode_avg_reward, episode_num_its



def get_epsilon_greedy_action(Q, current_state, num_actions, epsilon, env):

    if np.random.random() < epsilon:
        action = np.random.randint(0, num_actions)

        # mode = np.random.choice([0, 1, 2])
        # pos = np.array(env.get_target_location())
        # xy_index = env.state_to_index(pos)
        # action = 3*245 + mode
    else:
        # if there are multiple actions with the same value, choose one of them randomly
        best_actions = np.argwhere(Q[current_state[0], current_state[1], :] == np.max(
            Q[current_state[0], current_state[1], :])).flatten()
        action = np.random.choice(best_actions)

    return action


def policy(Q, grid_size):
    policy = np.chararray((grid_size, grid_size), itemsize=100, unicode=True)
    for row in range(0, grid_size):
        for col in range(0, grid_size):
            best_action = np.argmax(Q[row, col, :])
            policy[row, col] = f'{_env_action(best_action, grid_size)}'
    return policy

def _env_action(action, env):
    mode = action % 3
    index = np.floor(action/3)
    x = (index % env.num_blocks_x) * env.discretize + env.lower_xy_bounds[0]
    y = (index % env.num_blocks_y) * env.discretize - (env.upper_xy_bounds[1] - env.lower_xy_bounds[1])/2
    pos = [x, y]
    return {"mode": mode, "pos": pos}
