# Define Hyper-parameters
# Neural network
batch_size = 32
learning_rate = 0.003
lr_decay_active = True
lr_decay = 10000

# Q-Learning
# memory_size = 100
memory_size = 1000000
gamma = 0.99
num_episodes = 100000
save_memory = False

# target network update/copy weights average FIX
target_update_average_active = False
target_update_average_tau = 0.01

# Epsilon-greedy
eps_start = 1.0
eps_end = 0.1
eps_decay = 1000000

# Frequencies
# Policy network copied to target network
target_network_freq = 20
# Saving model
saving_model_freq = 50

# Type of rl
rl = 'DQN' # DQN or DDQN

# environment (atari)
environ_atari = 'Breakout-v0'
save_rewards_history = True
render_env = False

# CNN arch
CNN_arch = 'DQN' # 'RESNET20' or 'DQN'


# !python3 content/drive/My\ Drive/Colab\ Notebooks/rl-sandbox-repo/atari/learning/template.py --save_instance colab
