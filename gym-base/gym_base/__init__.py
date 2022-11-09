from gym.envs.registration import register

register(
    id='gym_base/GridWorld-v0',
    entry_point='gym_base.envs:GridWorldEnv',
    max_episode_steps=300,
)