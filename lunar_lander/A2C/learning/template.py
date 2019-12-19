import torch
import torch.nn as nn
from argparse import ArgumentParser
import math
import sys
import time
import os
import copy
import numpy as np
from parameters import *
from agent import Agent
sys.path.append('../neural_network/')
from neural_network import *
import gym
from skimage.transform import rescale, resize, downscale_local_mean

resume_episode = 0
savedir = ''
resumedir = ''

def initialize(args):
    global savedir, resumedir, resume_episode

    # directory management
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

    # Variable env contains the environment class
    env = gym.envs.make(env_name)

    # Define policy network
    actor = Actor(actor_learning_rate, actor_lr_decay, 8, 4).to(device).float()
    critic = Critic(critic_learning_rate, critic_lr_decay, 8, 1).to(device).float()
    agent = Agent(actor, critic, gamma, device)

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
    env, agent = initialize(args)

    starting_episode = 1
    if resume_episode != 0:
        starting_episode = resume_episode + 1

    for episode in range(starting_episode,num_episodes):

        state = env.reset()
        done = False

        if render_environment:
            env.render()

        steps = 0
        rewards = []
        to_avg_rewards = []
        values = []
        log_probs = []
        episode_score = 0

        while(not done):
            steps += 1

            if render_environment:
                env.render()

            action, log_prob = agent.select_action(state, env)
            next_state, reward, done, _ = env.step(action)

            episode_score += reward

            value = agent.estimate_value(state)
            print('\nValue: {}'.format(value.item()))
            next_value = agent.estimate_value(next_state)
            target = (reward + agent.gamma * next_value)
            advantage = target - value
            print('Eps: {}'.format(episode))
            print('Step: {}'.format(steps))
            print('In-step Adv: {}\n'.format(advantage.item()))

            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)
            to_avg_rewards.append(reward)
            
            # monte carlo like optimization
            if steps % episode_steps == 0 or done:
                agent.optimize(rewards, values, log_probs, next_value)
                rewards = []
                values = []
                log_probs = []

            state = next_state

        if actor_lr_decay_active:
            agent.actor.learning_rate_decay(episode)
        if critic_lr_decay_active:
            agent.critic.learning_rate_decay(episode)

        avg_reward = np.mean(to_avg_rewards)
        agent.rewards_history.append(avg_reward)
        agent.episode_score_history.append(episode_score)

        if save_model and episode % save_model_freq == 0:
            agent.save_data(episode, savedir)

        print('\nSteps: {}'.format(steps))
        print('Episode: {}'.format(episode))
        print('Episode lenght: {}'.format(steps))
        print('average reward: {}'.format(avg_reward))
        print('episode score:: {}'.format(episode_score))
        print('Actor learning rate: {}'.format(agent.actor.optimizer.param_groups[0]['lr']))
        print('Critic learning rate: {}\n'.format(agent.critic.optimizer.param_groups[0]['lr']))
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--resume')
    parser.add_argument('--save_instance', required=True)
    train(parser.parse_args())
