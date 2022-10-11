import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class BERTNetwork(nn.Module):

    def __init__(self, n_classes = 1):
        super(BERTNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        
        # freezing bert weights
        for param in self.bert.parameters():
          param.requires_grad = False

        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        out = self.drop(out['pooler_output'])
        out = self.linear(out)
        out = self.sigmoid(out)

        return out

