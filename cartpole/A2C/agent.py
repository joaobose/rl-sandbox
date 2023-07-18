import torch
import numpy as np
import os

from torch.distributions import Categorical

import parameters


class Agent():
    def __init__(self, actor, critic, gamma, device):
        # Network to train
        self.actor = actor
        self.critic = critic

        # Discount for future rewards
        self.gamma = gamma

        self.device = device
        self.steps_done = 0

        self.rewards_history = []
        self.episode_score_history = []

    def select_action(self, state, env):
        self.steps_done += 1

        state = torch.tensor([state]).float()

        if str(self.device) == 'cuda':
            state = state.cuda()
            out = self.actor(state)
        else:
            out = self.actor(state)

        print('Prob: {}'.format(out.detach()))

        m = Categorical(out)

        action = m.sample()

        log_prob = m.log_prob(action)

        return action.item(), log_prob

    def estimate_value(self, state):
        state = torch.tensor([state]).float()

        if str(self.device) == 'cuda':
            state = state.cuda()
            out = self.critic(state)
        else:
            out = self.critic(state)

        return out.squeeze(0)

    def optimize(self, rewards, values, log_probs, next_value):

        # Compute Q-values (target)
        Q_vals = np.zeros_like(rewards)
        t = len(rewards) - 1
        next_value = next_value.cpu().detach().numpy().squeeze(0)
        Q_vals[t] = rewards[t] + self.gamma * next_value

        t -= 1
        while t >= 0:
            Q_vals[t] = rewards[t] + self.gamma * Q_vals[t + 1]
            t -= 1

        values = torch.stack(values)
        Q_vals = torch.FloatTensor(Q_vals)
        log_probs = torch.stack(log_probs)

        if str(self.device) == 'cuda':
            values = values.cuda()
            Q_vals = Q_vals.cuda()

        advantage = Q_vals - values
        actor_loss = -log_probs * advantage.detach()
        critic_loss = 0.5 * advantage.pow(2)

        # Backpropagate networks
        self.actor.backpropagate(actor_loss, parameters.mean_loss)
        self.critic.backpropagate(critic_loss, parameters.mean_loss)

    def resume(self, checkpoint, is_copy):
        self.steps_done = checkpoint['steps']
        if is_copy:
            self.rewards_history = checkpoint['rewards_history']
            self.episode_score_history = checkpoint['episode_score_history']

        self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.actor.learning_rate = checkpoint['actor_learning_rate']
        self.actor.load_state_dict(checkpoint['actor_weights'])

        self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.critic.learning_rate = checkpoint['critic_learning_rate']
        self.critic.load_state_dict(checkpoint['critic_weights'])

    def save_data(self, episode, savedir):
        last_savedir = savedir
        last_savedir += '/agent_model'
        checkpoint_file = last_savedir + '/checkpoint.pth.tar'
        rewards_history = []
        episode_score_history = []

        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            rewards_history = checkpoint['rewards_history']
            episode_score_history = checkpoint['episode_score_history']

            if episode == 0:
                rewards_history = []
                episode_score_history = []

        rewards_history += self.rewards_history
        episode_score_history += self.episode_score_history

        data = {
            'steps': self.steps_done,
            'episode': episode,
            'critic_optimizer': self.critic.optimizer.state_dict(),
            'critic_learning_rate': self.critic.learning_rate,
            'critic_weights': self.critic.state_dict(),
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'actor_learning_rate': self.actor.learning_rate,
            'actor_weights': self.actor.state_dict(),
            'rewards_history': rewards_history,
            'episode_score_history': episode_score_history
        }
        self.low_save(data, savedir)
        self.rewards_history = []
        self.episode_score_history = []

    def low_save(self, data, savedir):
        savedir += '/agent_model'
        checkpoint_file = savedir + '/checkpoint.pth.tar'
        torch.save(data, checkpoint_file)
