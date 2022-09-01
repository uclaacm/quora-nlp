import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# setting path
sys.path.append('../')
from constants import *

class BERTNetwork(torch.nn.Module):
    """
    
    """

    def __init__(self, hidden1, hidden2):
        super(BERTNetwork, self).__init__()
        return NotImplementedError

    def forward(self, x):
        return NotImplementedError