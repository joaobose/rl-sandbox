import torch
import matplotlib.pyplot as plt
import parameters
from argparse import ArgumentParser
import os
import numpy as np

plt.rcParams["figure.figsize"] = (11.8,7.5)
# print(plt.rcParams["figure.figsize"])

parser = ArgumentParser()
parser.add_argument('--instance', required=True)
args = parser.parse_args()
dir = '../instances/{}'.format(args.instance)
assert os.path.exists(dir) , 'Resume directory not found'


model_file = os.path.join(dir, 'agent_model')
model_file = os.path.join(model_file, 'checkpoint.pth.tar')
assert os.path.exists(model_file), 'Error: model file not fund in folder'
if(torch.cuda.is_available()):
    model_data = torch.load(model_file)
else:
    model_data = torch.load(model_file, map_location='cpu')

# Plot avg rewards
raw_rewards_history = model_data['rewards_history']

# using exponentially weighted averages for rewards
if parameters.rewards_exp_avg:
    last_r = 0
    rewards_history = []
    for x in raw_rewards_history:
        r = parameters.rewards_betta * last_r + (1 - parameters.rewards_betta) * x
        rewards_history.append(r)
        last_r = x
else:
    rewards_history = raw_rewards_history

plt.plot(rewards_history, label="agent reward")

legend = plt.legend(loc='upper center', shadow=True)
plt.title('avg reward over episodes')
plt.ylabel('avg reward')
plt.xlabel('Episodes')
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

# Plot episode score
raw_episode_score_history = model_data['episode_score_history']

# using exponentially weighted averages for rewards
if parameters.score_exp_avg:
    last_s = 0
    episode_score_history = []
    for x in raw_episode_score_history:
        s = parameters.score_betta * last_s + (1 - parameters.score_betta) * x
        episode_score_history.append(s)
        last_s = x
else:
    episode_score_history = raw_episode_score_history

plt.plot(episode_score_history, label="Episode score")

legend = plt.legend(loc='upper center', shadow=True)
plt.title('Score over episodes')
plt.ylabel('Score')
plt.xlabel('Episodes')
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


