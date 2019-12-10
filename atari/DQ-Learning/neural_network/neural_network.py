import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DQN(nn.Module):
    def __init__(self, batch_size, learning_rate, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(32)

        linear_input_size = 12 * 9 * 32
        self.fc3 = nn.Linear(linear_input_size, 256)
        self.fc4 = nn.Linear(256, outputs)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)

    # Called with either one element to determine next action, or a batch
    def forward(self, x):
        # normalizing input
        x /= 255

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.fc3(x.view(x.size(0), -1)))
        x = F.relu(self.fc4(x))

        return x

    def backpropagate(self, y_pred, y):
        # loss = F.smooth_l1_loss(y_pred, y)
        loss = F.mse_loss(y,y_pred)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learning_rate_decay(self,episode,decay):
        lr = self.learning_rate * math.exp(-episode / decay)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
