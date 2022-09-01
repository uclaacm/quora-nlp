import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# setting path
sys.path.append('../')
from constants import *

class GPTNetwork(torch.nn.Module):
    """
    A bag of words neural network that transforms the vocab size into a target number.
    """

    def __init__(self, hidden1, hidden2):
        super(GPTNetwork, self).__init__()
        return NotImplementedError

    def forward(self, x):
        return NotImplementedError