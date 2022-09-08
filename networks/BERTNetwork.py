import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# setting path
sys.path.append('../')
from constants import *

class BERTNetwork(torch.nn.Module):
    def __init__(self):
        super(BERTNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        # freezing bert weights
        for param in self.bert.parameters():
          param.requires_grad = False

        self.drop = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
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