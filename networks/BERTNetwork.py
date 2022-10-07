import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import sys
# setting path
sys.path.append('../')
from constants import *

class BERTNetwork(torch.nn.Module):
    """
    BERT SESAME STREET HAHAHA
    """

    def __init__(self, vocab_size = 758, hidden1 = 128, hidden2 = 64, hidden3 = 8):
        super(BERTNetwork, self).__init__()
        self.BERT = SentenceTransformer("bert-base-nli-mean-tokens")
        
        # this logistic regression classifier could be replaced (ie. random forest or XDG or whatever else)
        self.fc1 = nn.Linear(vocab_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)

    def forward(self, x):
        with torch.no_grad:
            x = BERT.encode(x)
        x = F.relu(self.fc1(inputs.squeeze(1).float()))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

