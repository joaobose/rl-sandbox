import torch
import torch.nn as nn
import math
from argparse import ArgumentParser
import sys
import time
import os
import copy
import numpy as np
from parameters import *
from replay_memory import *
from agent import Agent
from epsilon_greedy import EpsilonGreedy
from q_values import QValues
sys.path.append('../neural_network/')
from resnet import *
from neural_network import *
import gym
from frames import *

resume_episode = 0
savedir = ''
resumedir = ''

def initialize(args):
    global savedir, resumedir, resume_episode

    savedir = '../instances'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    savedir = '../instances/{}'.format(args.save_instance)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        os.makedirs(savedir + '/agent_model')

    if args.resume is not None:
        resumedir = '../instances/{}'.format(args.resume)
        assert os.path.exists(resumedir) , 'resume directory not found'

    # Define PyTorch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Variable env contains the environment class (python game)
    env = gym.envs.make(environ_atari)

    if CNN_arch == 'RESNET20':
        # Define policy and target networks
        policy_net = resnet20(batch_size, learning_rate,env).to(device).float()
        target_net = resnet20(batch_size, learning_rate,env).to(device).float()
    elif CNN_arch == 'DQN':
        policy_net = DQN(batch_size, learning_rate, env.action_space.n).to(device).float()
        target_net = DQN(batch_size, learning_rate, env.action_space.n).to(device).float()


    # Copy the weights
    target_net.load_state_dict(policy_net.state_dict())
    # Do not backpropagate target network
    target_net.eval()

    memory = ReplayMemory(memory_size)
    strategy = EpsilonGreedy(eps_start, eps_end, eps_decay)
    agent = Agent(policy_net, target_net, memory, strategy, gamma, 3, device)

    # Resume
    if args.resume is not None:
        checkpoint_file = os.path.join(resumedir,'agent_model')
        checkpoint_file = os.path.join(checkpoint_file,'checkpoint.pth.tar')
        assert os.path.exists(checkpoint_file), 'Error: resume option was used but checkpoint was not found in the folder'
        checkpoint = torch.load(checkpoint_file)
        agent.resume(checkpoint,resumedir != savedir)
        resume_episode = checkpoint['episode']

    return env, agent

def train(args):
    global savedir
    env, agent = initialize(args)

    starting_episode = 1
    if resume_episode != 0:
        starting_episode = resume_episode + 1

    for episode in range(starting_episode,num_episodes):
        start = time.time()
        state = env.reset()
        state = np.transpose(state, (2, 0, 1))
        done = False
        frames = []
        rewards = []
        add_frame(state,frames)

        if render_env:
            env.render()

        # Loop through steps of episode
        while(not done):

            if render_env:
                env.render()

            # Select and perform an action
            if len(frames) == 3:
                action = agent.select_action(concat_frames(frames), env)
            else:
                action = agent.select_action(None, env, True)

            # state, action, reward and next_state
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)

            next_state = np.transpose(next_state, (2, 0, 1))

            action = torch.tensor([[action]])
            reward = torch.tensor([reward])

            # Store the experience in memory
            if len(frames) == 3:
                agent.memory.push(Experience(concat_frames(frames), action, reward, concat_next_frames(next_state,frames)))

            # Move to the next state
            state = next_state
            add_frame(state,frames)
            # Perform one step of the optimization (on the policy network)
            agent.optimize_policy()

        # Update the target network, copying all weights and biases in the target network
        if agent.steps_done % target_network_freq == 0:
            agent.update_target_net()

        # Save agents
        if episode % saving_model_freq == 0:
            agent.save_data(episode, savedir)

        # learing rate decay
        if lr_decay_active:
            agent.policy_net.learning_rate_decay(episode, lr_decay)

        if save_rewards_history:
            agent.points_history.append(np.mean(rewards))

        print('Episode: {} | Time: {}'.format(episode, round(time.time() - start, 2)))
        print('Steps done: {}'.format(agent.steps_done))
        print('Eps threshold: {}'.format(agent.strategy.get_exploration_rate(agent.steps_done)))
        print('learning_rate: {}'.format(agent.policy_net.optimizer.param_groups[0]['lr']))
        print('average reward: {}\n'.format(np.mean(rewards)))

    print('Done')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--save_instance', required=True)
    train(parser.parse_args())
