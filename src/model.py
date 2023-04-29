import transformers
from transformers import BertModel
import torch
from torch import nn
import random as rn
import numpy as np
np.random.seed(17)
rn.seed(12345)

class SentimentCheck(nn.Module):
    
    def __init__(self, n_classes=3):
        super(SentimentCheck, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.45)
        self.fc = nn.Linear(self.model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        _, output = self.model(input_ids = input_ids, attention_mask = attention_mask)[0:2]
        output = self.drop(output)
        output = self.fc(output)
        return output