# Define Hyper-parameters
batch_size = 1

# Actor
actor_learning_rate = 0.001
actor_lr_decay_active = False
actor_lr_decay = 10000

# Critic
critic_learning_rate = 0.001
critic_lr_decay_active = False
critic_lr_decay = 10000

# algorithm
gamma = 0.8
num_episodes = 100000
episode_steps = 50  # N
mean_loss = False

# environment
env_name = 'CartPole-v1'
render_environment = True

# model checkpoint
save_model = True
save_model_freq = 10

# plots
rewards_exp_avg = False
rewards_betta = 0.99
score_exp_avg = True
score_betta = 0.8
