import torch
import torch.nn as nn


import torch.nn.functional as F

class StartingNetwork(torch.nn.Module):
    """
    A neural network that utilizes the glove embeddings.
    """

    def __init__(self, vocab_size, hidden1, hidden2):
        return NotImplementedError
        # TODO: Create a network that uses the GloVe embeddings
        # super(StartingNetwork, self).__init__()
        # self.fc1 = nn.Linear(vocab_size, hidden1)
        # self.fc2 = nn.Linear(hidden1, hidden2)
        # self.fc3 = nn.Linear(hidden2, 1)

    def forward(self, x):
        # TODO: Use activation functions to pass data through the NN layers
        # x = F.relu(self.fc1(x.squeeze(1).float()))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        return x