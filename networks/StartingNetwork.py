import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# setting path
sys.path.append('../')
from constants import *

class StartingNetwork(torch.nn.Module):
    """
    A bag of words neural network that transforms the vocab size into a target number.
    """

    def __init__(self, hidden1, hidden2):
        super(StartingNetwork, self).__init__()
        self.fc1 = nn.Linear(MAX_SEQ_LEN, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x.squeeze(1).float()))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x