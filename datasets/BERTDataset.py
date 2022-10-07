import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

class BERTDataset(torch.utils.data.Dataset):

    """
    Dataset that contains (sequences and targets)
    
    path should contain path to the appropriate csv (either train.csv, test.csv)

    """

    def __init__(self, csv_path, args, is_train = True):

        self.BERT = SentenceTransformer("bert-base-nli-mean-tokens")

        df = pd.read_csv(csv_path)

        train, test = train_test_split(df, test_size=args.split_percent/100, random_state=69420)
        self.df = train if is_train else test
        self.df.reset_index(drop = True, inplace = True)
        self.len = len(self.df)

        self.qs = self.df.question_text

        # self.ls = np.asarray(self.df['target'].values, dtype='int')

        
    def __getitem__(self, index):
 
        # comment = self.df.loc[index, 'question_text']
        comment = self.qs[index]
        label = self.df.loc[index, 'target']

        return self.BERT.encode(comment), label


    def __len__(self):
        return self.len
