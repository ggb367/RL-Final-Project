from random import betavariate
import numpy as np


def q_learning(env):
    alpha = 0.5
    epsilon = 0.2
    gamma = 0.95

    num_of_episodes = 10000
    number_of_iterations = 100
    num_of_actions = 4
    num_of_rows = 4
    num_of_columns = 4


    Q = np.zeros((num_of_rows, num_of_columns, num_of_actions))

    for _ in range(num_of_episodes):
        observation, _ = env.reset()
        current_state = observation['agent']
        
        for _ in range(number_of_iterations):
            action = get_epsilon_greedy_action(Q, current_state, num_of_actions, epsilon)
            observation, reward, terminated, _, _ = env.step(action)

            next_state = observation['agent']

            Q[current_state[0], current_state[1], action] += alpha * (
                reward + gamma * np.max(Q[next_state[0], next_state[1], :]) - Q[current_state[0], current_state[1], action])
            
            current_state = next_state
            if terminated:
                break
        
        # print(Q[3, 1])
        # print("---")
    return Q


def get_epsilon_greedy_action(Q, current_state, num_actions, epsilon):
    if np.random.random() < epsilon:
        action = np.random.randint(0, num_actions)
    else:
        action = np.argmax(Q[current_state[0], current_state[1], :])

    return action


def policy(Q):
    action_to_direction = {
        0: 'R',  # Right
        1: 'U',  # Up
        2: 'L',  # Left
        3: 'D',  # Down
    }
    policy = np.chararray((4, 4), unicode=True)
    for row in range(0, 4):
        for col in range(0, 4):
            best_action = np.argmax(Q[row, col, :])
            policy[row, col] = action_to_direction[best_action]

    return policy
