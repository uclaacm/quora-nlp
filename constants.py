"""
Add hyperparams here pls
"""

LOCAL = True

# How many times should the model pass through the training set?
EPOCHS = 3
# EPOCHS = 100

# How many comments to use in one training iteration?
BATCH_SIZE = 30

# How often should we evaluate the model (in iterations)?
N_EVAL = 25

# What percent of data to use for test?
SPLIT_PERC = 20 

# How much should the gradient descent move?
LEARNING_RATE = 5e-3
CLASS_RATIO = 2

# Where is the <> dir?
DATA_DIR = ""

# Hyperparameters for SequencesCountVectorizer
MAX_SEQ_LEN = 25
MIN_FREQ = 0.00
MAX_FREQ = 0.98
