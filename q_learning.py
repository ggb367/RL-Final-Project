# from random import betavariate
import numpy as np


def q_learning(env):
    alpha = 0.5
    epsilon = 0.2
    gamma = 0.95

    num_of_episodes = 100
    number_of_iterations = 100

    num_of_rows = env.size
    num_of_columns = env.size

    # There are 3 modes, which each can have (grid_size, grid_size) poses
    num_of_actions = 3 * env.size * env.size

    Q = np.zeros((num_of_rows, num_of_columns, num_of_actions))

    for _ in range(num_of_episodes):
        observation, _ = env.reset()
        current_state = observation['object']

        for _ in range(number_of_iterations):
            action = get_epsilon_greedy_action(
                Q, current_state, num_of_actions, epsilon)

            env_action = _env_action(action, env.size)
            observation, reward, terminated, _, _ = env.step(env_action)

            next_state = observation['object']

            Q[current_state[0], current_state[1], action] += alpha * (reward + gamma * np.max(
                Q[next_state[0], next_state[1], :]) - Q[current_state[0], current_state[1], action])

            current_state = next_state
            if terminated:
                break
    return Q


def get_epsilon_greedy_action(Q, current_state, num_actions, epsilon):
    if np.random.random() < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(Q[current_state[0], current_state[1], :])

    return action


def policy(Q, grid_size):
    policy = np.chararray((grid_size, grid_size), itemsize=100, unicode=True)
    for row in range(0, grid_size):
        for col in range(0, grid_size):
            best_action = np.argmax(Q[row, col, :])
            policy[row, col] = f'{_env_action(best_action, grid_size)}'
    return policy


def _env_action(action, grid_size):
    mode = action % 3
    index = np.floor(action/3)
    pos = (int(index // grid_size), int(index % grid_size))
    return {"mode": mode, "pos": pos}
