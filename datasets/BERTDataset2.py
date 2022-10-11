from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class BERTDataset(Dataset):

    def __init__(self, targets, tokenizer, max_len):
        df = pd.read_csv(csv_path)

        train, test = train_test_split(df, test_size=args.split_percent/100, random_state=69420)
        self.df = train if is_train else test
        self.df.reset_index(drop = True, inplace = True)

        self.question_text = df['question_text']
        self.targets = df['']
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.question_text)
  
    def __getitem__(self, item):
        text = str(self.question_text[item])
        label = self.df.loc[item, 'target']

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return {
          'question_text': text,
          'inputs': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'target': torch.tensor(target, dtype=torch.long)
        }