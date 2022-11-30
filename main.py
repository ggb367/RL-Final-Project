import gym
import gym_base
from q_learning import q_learning, policy


grid_env = gym.make('gym_base/GridWorld-v0')

# grid_env.display_scenario()

Q = q_learning(grid_env)
p = policy(Q, grid_env.size)

for row in range(0, 4):
    for col in range(0, 4):
        print(f'state: ({row}, {col}), p: {p[row, col]}')

# grid_env.reset()

# action = {"mode": 2, "pos": (, 0)}

# observation, reward, terminated, _, _ = grid_env.step(action=action)
# grid_env.display_scenario()
