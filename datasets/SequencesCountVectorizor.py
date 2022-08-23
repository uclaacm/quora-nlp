from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd 

class SequencesCountVectorizer(Dataset):
    def __init__(self, path, max_seq_len, min_freq, max_freq, class_ratio=1, is_train=True):
        """
        path - the path of the training csv file
        max_seq_len - the maximum question length we want to consider (H) TODO: what does this H mean
        min_freq - the minimum frequency in the dataset (in percentage for words to be kept (H))
        class_ratio - the L + ratio of the 0 label to 1 label in training data
        """

        self.max_seq_len = max_seq_len
        df = pd.read_csv(path)
        df = df.dropna() # remove any rows with erroneous variable types
       
        # calculations for data augmentation
        total = len(df)
        num_zero = len(df[df['target'] == 0])
        num_one = total - num_zero
        cur_ratio = num_zero/num_one
    
        # fraction of current 0s we want to retain
        retention_ratio = class_ratio / cur_ratio

        if retention_ratio < 1 and is_train:
          df = (df).drop((df).query('target < 1').sample(frac=(1-retention_ratio)).index)

        print(len(df))
        train, test = train_test_split(df, test_size=0.2)
        df = train if is_train else test
        # create a vectorizer object
        # only keep words with >15% frequency this is a hyper-param
        vectorizer = CountVectorizer(stop_words='english', min_df=min_freq, 
                                     max_df=max_freq) 

        questions_list = df.question_text.tolist()

        # fit the vectorizer on our corpus vocabulary
        vectorizer.fit(questions_list)
        
        # save a dictionary of our corpus vocabulary
        self.token2idx = vectorizer.vocabulary_
        print(self.token2idx)
        self.token2idx['<PAD>'] = max(self.token2idx.values()) + 1 #what is this? whitespace?
        print(self.token2idx['<PAD>'])

        # create a tokenizer object to 
        tokenizer = vectorizer.build_analyzer()
        
        # generate encoding for each token in the vocabulary (this is bag of 
        # words)
        self.encode = lambda x: [self.token2idx[token] for token in tokenizer(x)
                                 if token in self.token2idx]
        
        # this is a padder helper function that helps pad all sequences to a 
        # uniform length
        self.pad = lambda x: x + (max_seq_len - len(x)) * [self.token2idx['<PAD>']]
        
        # encode every question in the dataframe and save the encoding in 
        # sequences
        sequences = [self.encode(sequence)[:max_seq_len] 
                     for sequence in df.question_text.tolist()]

        # converting the sequence and label columns into iterables
        sequences, self.labels = zip(*[(sequence, label) for sequence, label
                                    in zip(sequences, df.target.tolist()) 
                                    if sequence])
        
        # finally padding all sequences as the last step in the dataset 
        # self.sequences contains the processed text and self.labels the labels
        self.sequences = [self.pad(sequence) for sequence in sequences]

    def __getitem__(self, i):
        assert len(self.sequences[i]) == self.max_seq_len
        return self.sequences[i], self.labels[i]
    
    def __len__(self):
        return len(self.sequences)