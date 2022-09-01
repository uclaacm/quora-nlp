import torch
import torch.nn as nn


import torch.nn.functional as F

class EmbeddingNetwork(torch.nn.Module):
    """
    A neural network that utilizes the glove embeddings.
    Confirm that Advit should complete this
    """

    def __init__(self, vocab_size, hidden1, hidden2):  
        # TODO: Create a network that uses the GloVe embeddings
        super(EmbeddingNetwork, self).__init__()
        return NotImplementedError

    def forward(self, x):
        # TODO: Use activation functions to pass data through the NN layers
        return NotImplementedError