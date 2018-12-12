import gym

env = gym.make('Acrobot-v1')
env.reset()
for _ in range(100):
    env.render()
    env.step(env.action_space.sample())