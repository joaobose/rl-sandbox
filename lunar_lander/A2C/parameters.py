# Define Hyper-parameters
batch_size = 1

# Actor
actor_learning_rate = 0.001
actor_lr_decay_active = True
actor_lr_decay = 8000

# Critic
critic_learning_rate = 0.001
critic_lr_decay_active = True
critic_lr_decay = 8000

# algorithm
gamma = 0.9995
num_episodes = 100000
episode_steps = 50  # N
mean_loss = False

# environment
env_name = 'LunarLander-v2'
render_environment = False

# model checkpoint
save_model = True
save_model_freq = 10

# plots
rewards_exp_avg = False
rewards_betta = 0.99
score_exp_avg = True
score_betta = 0.999
