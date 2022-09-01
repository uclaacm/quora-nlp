import torch, re
import numpy as np
import pandas as pd


class StartingDataset(torch.utils.data.Dataset):

    """
    Dataset that contains (sequences and targets)
    
    path should contain path to the appropriate csv (either train.csv, test.csv)

    """

    def __init__(self, csv_path, glove_path):

        self.df = pd.read_csv(path)
        self.len = len(self.df)

        self.qs = np.asarray(self.df['question_text'].values)
        self.ls = np.asarray(self.df['target'].values, dtype='int')

        # print(f"Number of loaded comments : {self.len}")
        # print(f"Number of sincere Q (0)   : {np.count_nonzero(self.ls == 0)}")
        # print(f"Number of insincere Qs (1): {np.count_nonzero(self.ls == 1)}")


        # For Bag of Words Approach 

        # print(f"Beginning unique word count (for bag of words)...")

        # self.wordset = set()
        # for comment in self.qs: 
        #     for word in (re.sub(r"[^a-zA-Z0-9]", " ", comment.lower()).split()):
        #         self.wordset.add(word) 
        
        # print(f"Number of unique words    : {len(self.wordset)}")


        # For GLOVE Embeddings Approach 

        self.glove_embeddings_path = glove_path
        self.glove_embeddings = self.gen_glove_embedding_dict()

        
    def __getitem__(self, index):
 
        comment = self.qs[index]
        label = self.ls[index]

        # print(f"(__getitem__) > Comment: {comment}")
        # print(f"(__getitem__) >   Label: {label}")

        #embedding = self.bag_of_words(comment)
        embedding = self.get_glove_embeddings(comment)

        # print(f"(__getitem__) >   GLOVE: {embedding}")

        return comment, label, embedding

    def gen_glove_embedding_dict(self): 

        f = open(self.glove_embeddings_path, "r", encoding="utf-8")

        glove = {} 
        for line in f.readlines(): 
            raw_line = line.split(" ") 
            word, embedding = raw_line[0], raw_line[1:]
            
            # print(f"(gen_glove_embedding_dict) > Word: {word}, Embedding: {embedding}")
            glove[word] = np.asarray(embedding, dtype='float')
        
        f.close() 
        return glove 
    

    def bag_of_words(self, comment): 

        # Sourced from: https://www.analyticsvidhya.com/blog/2021/08/a-friendly-guide-to-nlp-bag-of-words-with-python-example/

        tf_diz = dict.fromkeys(self.wordset,0)

        for word in (re.sub(r"[^a-zA-Z0-9]", " ", comment.lower()).split()):
            tf_diz[word]=comment.count(word)

        return tf_diz


    def __len__(self):
        return self.len
