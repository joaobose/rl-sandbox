import gym

env = gym.make('Breakout-v0')
# env = gym.wrappers.Monitor(env, './video/', force = True)

for i_episode in range(200000):
    observation = env.reset()
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.close()
