"""
Main function used to train the Quora Insincere Comments model. 
Utilizes file constants.py as well as the following hyperparameters: 

- Epochs - how many times should the model pass through the dataset? 
- Batch size - how many comments to use in one training iteration? 
- n_eval - how often should we evaluate the model (in iterations)? 
- datadir - where is the Quora Insincere Comments dataset (6 GB)? 

TODO: Add more descriptions for usage here!

"""

import argparse
import os, torch 

# TODO: Change these inports to only import models/dataset as specified by args 

from data.StartingDataset import StartingDataset 
from networks.StartingNetwork import StartingNetwork
from training.starting_train import starting_train


def main(): 

    args = parse_arguments() 

    # Create our model, and begin starting_train()

    model = StartingNetwork()
    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=args.n_eval,
    )



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

    parser.add_argument("--epochs", type=int, default=constants.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=constants.BATCH_SIZE)
    parser.add_argument("--n_eval", type=int, default=constants.N_EVAL)
    parser.add_argument("--datadir", type=str, default=constants.DATA_DIR)

    return parser.parse_args()


if __name__ == "__main__":
    main()