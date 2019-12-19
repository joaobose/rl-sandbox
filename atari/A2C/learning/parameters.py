# Define Hyper-parameters
batch_size = 1

# Actor
actor_learning_rate = 0.001
actor_lr_decay_active = True
actor_lr_decay = 10000
argmax_deterministic_actor = False # True doesn't work

# Critic
critic_learning_rate = 0.001
critic_lr_decay_active = True
critic_lr_decay = 10000

# algorithm
gamma = 0.9995
num_episodes = 100000
episode_steps = 50 # N
mean_loss = False # !

# environment
env_name = 'Breakout-v4'
render_environment = True
eps_end_on_life = True
autoshoot_first = True

# model checkpoint
save_model = True
save_model_freq = 10

# plots
rewards_exp_avg = True
rewards_betta = 0.6
score_exp_avg = True
score_betta = 0.6