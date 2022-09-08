from transformers import BertTokenizer

class BERTDataset(Dataset):

    def __init__(self, path, max_seq_len, class_ratio=1, is_train=True):
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

        train, test = train_test_split(df, test_size=0.2)
        df = train if is_train else test

        self.question_text = df.question_text.tolist()
        self.targets = df.target.tolist()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')



    def __len__(self):
        return len(self.question_text)
  
    def __getitem__(self, item):
        text = str(self.question_text[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
          text,
          add_special_tokens=True,
          max_length=self.max_seq_len,
          return_token_type_ids=False,
          padding='max_length',
          truncation=True,
          return_attention_mask=True,
          return_tensors='pt',
        )

        return {
          'question_text': text,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
        }