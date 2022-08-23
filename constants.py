"""
Add hyperparams here pls
"""

LOCAL = True

# How many times should the model pass through the training set?
EPOCHS = 100

# How many comments to use in one training iteration?
BATCH_SIZE = 30

# How often should we evaluate the model (in iterations)?
N_EVAL = 25

# Where is the <> dir?
DATA_DIR = ""

# Hyperparameters for SequencesCountVectorizer
MAX_SEQ_LEN = 25
MIN_FREQ = 0.00
MAX_FREQ = 0.98
