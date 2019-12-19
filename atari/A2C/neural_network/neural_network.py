import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):
    def __init__(self, learning_rate, lr_decay, input, output):
        super(Actor, self).__init__()
        self.input = input
        self.output = output
        
        self.conv1 = nn.Conv2d(input, 16, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        linear_input_size = 12 * 9 * 32
        self.fc3 = nn.Linear(linear_input_size, 256)
        self.fc4 = nn.Linear(256, output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # normalizing input
        x /= 255

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        
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
        
        self.conv1 = nn.Conv2d(input, 16, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        linear_input_size = 12 * 9 * 32
        self.fc3 = nn.Linear(linear_input_size, 256)
        self.fc4 = nn.Linear(256, output)
        
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, x):
        # normalizing input
        x /= 255

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = self.fc4(x)
        
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