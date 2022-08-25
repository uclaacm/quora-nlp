"""
Main function used to train the Quora Insincere Comments model. 
Utilizes file py as well as the following hyperparameters: 

- Epochs - how many times should the model pass through the dataset? 
- Batch size - how many comments to use in one training iteration? 
- n_eval - how often should we evaluate the model (in iterations)? 
- datadir - where is the Quora Insincere Comments dataset (6 GB)? 

TODO: Add more descriptions for usage here!

"""

import argparse
import os, torch
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader

# TODO: Change these inports to only import models/dataset as specified by args 

from datasets.StartingDataset import StartingDataset 
from datasets.SequencesCountVectorizor import SequencesCountVectorizer
from networks.StartingNetwork import StartingNetwork
from training.train import train
from training.test import test_loop
from constants import *

def main():

    args = parse_arguments()

    # Init dataset
    data_path = "./data/" if LOCAL else "/kaggle/input/quora-insincere-questions-classification/"
    train_path = data_path + "train.csv"
    test_path = data_path + "test.csv" #this shit isn't labelled
    train_dataset = SequencesCountVectorizer(train_path, max_seq_len=MAX_SEQ_LEN, min_freq=MIN_FREQ, max_freq=MAX_FREQ, class_ratio=CLASS_RATIO)
    test_dataset = SequencesCountVectorizer(train_path, max_seq_len=MAX_SEQ_LEN, min_freq=MIN_FREQ, max_freq=MAX_FREQ, class_ratio=CLASS_RATIO, is_train=False)

    # enables GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # choose your model, loss function and optimizer
    model = StartingNetwork(vocab_size=len(train_dataset.token2idx), hidden1=128, hidden2=64)
    loss_criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training loop
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    train(model, train_loader, optimizer, loss_criterion, device)

    # testing loop
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate)
    test_loop(model, test_loader, device)


# this crap seems incompatible with the constants file we had
"""

Add supported arguments to be passed in from commandline. 
Currently, main function supports the following args: 

- epochs: <add desc here> 
- batch_size: <add desc here>  
- n_eval: <add desc here> 
- datadir: <add desc here> 

TODO: Add embeddingsdir, embeddingschoice, networkchoice 
TODO: Make sure to add defaults for each in constants.py

"""

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=N_EVAL)
    parser.add_argument("--datadir", type=str, default=DATA_DIR)

    return parser.parse_args()


def collate(batch): # dynamically pad sequences to the longest one during batch creation
    inputs = torch.LongTensor([item[0] for item in batch])
    target = torch.FloatTensor([item[1] for item in batch])
    return inputs, target

if __name__ == "__main__":
    main()