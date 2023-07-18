import gymnasium as gym

env = gym.make('Breakout-v0', render_mode='human')

for i_episode in range(100):
    observation = env.reset()

    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if t == 0:
            print('Observation shape:', observation.shape)

        if terminated or truncated:
            print('Episode finished after {} timesteps'.format(t+1))
            break

env.close()
