import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):
    def __init__(self, learning_rate, lr_decay, input, output):
        super(Actor, self).__init__()
        self.input = input
        self.output = output
        
        self.hidden1 = nn.Linear(self.input, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, self.output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x.view(x.size(0), -1))

        return F.softmax(x, dim=1)

    def backpropagate(self, actor_loss, mean_loss):
        if mean_loss:
            loss = actor_loss.mean()
        else:
            loss = actor_loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learning_rate_decay(self, episode):
        lr = self.learning_rate * math.exp(- episode / self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


class Critic(nn.Module):
    def __init__(self, learning_rate, lr_decay, input, output):
        super(Critic, self).__init__()
        
        assert input % 4 == 0
        self.input = input
        self.output = output
        
        self.hidden1 = nn.Linear(input, 32)
        self.hidden2 = nn.Linear(32, 64)
        self.hidden3 = nn.Linear(64, output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.hidden3(x.view(x.size(0), -1))
        
        return torch.tanh(x)

    def backpropagate(self, critic_loss, mean_loss):
        if mean_loss:
            loss = critic_loss.mean()
        else:
            loss = critic_loss.sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def learning_rate_decay(self, episode):
        lr = self.learning_rate * math.exp(- episode / self.lr_decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr