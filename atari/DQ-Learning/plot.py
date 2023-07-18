import torch
import matplotlib.pyplot as plt
from parameters import *
from argparse import ArgumentParser
import os
import numpy as np

plt.rcParams["figure.figsize"] = (11.8, 7.5)

parser = ArgumentParser()
parser.add_argument('--instance', required=True)
args = parser.parse_args()
dir = './instances/{}'.format(args.instance)
assert os.path.exists(dir), 'Resume directory not found'


model_file = os.path.join(dir, 'agent_model')
model_file = os.path.join(model_file, 'checkpoint.pth.tar')
assert os.path.exists(model_file), 'Error: model file not fund in folder'
if (torch.cuda.is_available()):
    model_data = torch.load(model_file)
else:
    model_data = torch.load(model_file, map_location='cpu')

plt.plot(model_data['points_history'], label="agent reward")

legend = plt.legend(loc='upper center', shadow=True)
plt.title('reward over episodes')
plt.ylabel('reward')
plt.xlabel('Episodes (x{})'.format(saving_model_freq))
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
