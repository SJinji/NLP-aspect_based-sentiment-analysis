import torch
import numpy as np
from torch.utils.data import Dataset
import random as rn
np.random.seed(17)
rn.seed(12345)


#----------------PREPROCESS RAW DATA----------------#
# Converting aspects to questions and consider sentences as replies (account for context)

aspect_map = {"AMBIENCE#GENERAL":"What do you think of the ambience ?",
              "FOOD#QUALITY":"What do you think of the quality of the food ?",
              "SERVICE#GENERAL":"What do you think of the service ?",
              "FOOD#STYLE_OPTIONS": "What do you think of the food choices ?",
              "DRINKS#QUALITY":"What do you think of the drinks?",
              'RESTAURANT#MISCELLANEOUS': "What do you think of the restaurant ?",
              'RESTAURANT#GENERAL': "What do you think of the restaurant ?",
              'LOCATION#GENERAL': 'What do you think of the location ?',
              'DRINKS#STYLE_OPTIONS': "What do you think of the drink choices ?",
              'RESTAURANT#PRICES':'What do you think of the price of it ?',
              'DRINKS#PRICES':'What do you think of the price of it ?',
              'FOOD#PRICES': 'What do you think of the price of it ?'            
              }
label_encode = {"negative":0,"neutral":1,"positive":2}

def preprocessing(file):
    file[0] = file[0].apply(lambda x: label_encode[x])
    file[1] = file[1].apply(lambda x: aspect_map[x])
    return file

#----------------PREPARE PYTORCH DATASET----------------#
# Function to calculate max token length as input for creating pytorch dataset
def max_len_token(file, tokenizer):
    token_lengths = []
    for sentence in file[4]:
        tokens = tokenizer.encode(sentence, max_length=1000)
        token_lengths.append(len(tokens))
    return max(token_lengths)

# Create pytorch dataset
class create_dataset(Dataset):
    
    def __init__(self, sentences, aspect_categories, target_terms, labels, tokenizer, max_len_token):
        
        self.sentences = sentences
        self.aspect_categories = aspect_categories
        self.target_terms = target_terms
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len_token = max_len_token
        
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        aspect_category = str(self.aspect_categories[index])
        target_term = str(self.target_terms[index])
        label = self.labels[index]

        aspect = aspect_category + ' ' + target_term
        
        # Get the attention maks
        encoding = self.tokenizer.encode_plus(
            text = sentence,
            text_pair = aspect,
            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
            max_length=self.max_len_token, # Pad & truncate all sentences.
            return_token_type_ids=False,
            pad_to_max_length=True, # Pad to the end of the sentence
            return_attention_mask=True, # Construct attn. masks.
            return_tensors='pt',  # Return PyTorch tensors
            truncation=True)

        return {
            'sentence_text': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)}

    
    