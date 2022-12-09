import pdb

import numpy as np

from multimodal_planning_v2 import PathPlanner


from tqdm import tqdm

def q_learning(env):
    learning_rate = 0.4
    epsilon = 0.4  # randomness factor
    discount_factor = 0.95

    num_of_episodes = 50
    number_of_iterations = 15

    num_of_rows = env.num_blocks_x
    num_of_columns = env.num_blocks_y

    # There are 3 modes, which each can have (grid_size, grid_size) poses
    num_of_actions = 3 * env.num_blocks_x * env.num_blocks_y

    Q = np.zeros((num_of_rows, num_of_columns, num_of_actions))

    episode_avg_reward = np.zeros(num_of_episodes)
    episode_num_its = np.zeros(num_of_episodes)

    for ep in tqdm(range(num_of_episodes)):
        observation, _ = env.reset(random_start=False)
        current_xy = observation['object']
        current_state = state_to_index(current_xy, env)
        avg_reward = 0
        for it in range(number_of_iterations):
            action = get_epsilon_greedy_action(
                Q, current_state, num_of_actions, epsilon)

            env_action = _env_action(action, env)
            observation, reward, terminated, _, _ = env.step(env_action)
            avg_reward += reward

            next_xy = observation['object']
            next_state = state_to_index(next_xy, env)

            Q[current_state[0], current_state[1], action] += learning_rate * (reward + discount_factor * np.max(
                Q[next_state[0], next_state[1], :]) - Q[current_state[0], current_state[1], action])


            current_state = next_state
            if terminated:
                episode_avg_reward[ep] = avg_reward / (it + 1)
                episode_num_its[ep] = it
                break
            if it == number_of_iterations - 1:
                episode_avg_reward[ep] = avg_reward / (it + 1)
                episode_num_its[ep] = it
    return Q, episode_avg_reward, episode_num_its


def state_to_index(state, env):
    interval = env.discritize
    x = state[0]
    y = state[1]
    x_index = int(x / interval)
    y_index = int(y / interval)
    # if x_index >= x_grid_size:
    #     x_index = x_grid_size - 1
    # if y_index >= y_grid_size:
    #     y_index = y_grid_size - 1
    return [x_index, y_index]



def get_epsilon_greedy_action(Q, current_state, num_actions, epsilon):
    if np.random.random() < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        # if there are multiple actions with the same value, choose one of them randomly
        # best_actions = np.argwhere(Q[current_state[0], current_state[1], :] == np.max(
        #     Q[current_state[0], current_state[1], :])).flatten()
        # action = np.random.choice(best_actions)
        # choose an action that is a multiple of 3, within len of Q
        action = np.random.randint(0, len(Q[0, 0, :]), 1)[0]
        action = action - (action % 3)

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
    interval = env.discritize
    x = (index % env.num_blocks_x) * interval
    y = np.floor(index / env.num_blocks_x) * interval
    pos = [x, y]
    pdb.set_trace()
    return {"mode": mode, "pos": pos}
