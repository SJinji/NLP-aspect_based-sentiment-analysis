import transformers
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F
import random as rn
from random import shuffle
np.random.seed(17)
rn.seed(12345)

from processing import *
from model import *



class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """

    ###############################################################################
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        
        #----------------LOAD FILE & PREPROCESS----------------#
        # Load the files into pandas dataframe
        train_file = pd.read_csv(train_filename, sep='\t', header=None)
        if dev_filename is not None:
            dev_file = pd.read_csv(dev_filename, sep='\t', header=None)
        
        # Proprocess the file
        self.trainfile = preprocessing(train_file)
        if dev_filename is not None:
            self.dev_file = preprocessing(dev_file)
        
        #----------------ENCODING & GET DATALOADER------------------#
        # Set the pretrained model & tokenizing
        PRE_TRAINED_MODEL = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)
        
        # Load the dataloader
        self.bs = 16 # Set batch size
        self.max_length = max_len_token(self.trainfile, self.tokenizer)
        self.train_dataset = create_dataset(
            sentences=self.trainfile[4].to_numpy(),
            aspect_categories = self.trainfile[1].to_numpy(),
            target_terms=self.trainfile[2].to_numpy(),
            labels = self.trainfile[0].to_numpy(),
            tokenizer = self.tokenizer,
            max_len_token = self.max_length)
        self.train_loader = DataLoader(self.train_dataset, batch_size= self.bs, shuffle=True)
        if dev_filename is not None: 
            self.dev_dataset = create_dataset(
                sentences=self.dev_file[4].to_numpy(),
                aspect_categories = self.dev_file[1].to_numpy(),
                target_terms=self.dev_file[2].to_numpy(),
                labels = self.dev_file[0].to_numpy(),
                tokenizer = self.tokenizer,
                max_len_token = self.max_length)
            self.val_loader = DataLoader(self.dev_dataset, batch_size= self.bs, shuffle=True)
            
        #----------------MODEL & HYPEPARAMETERS------------------#
        # Define the devide and the model
        self.device = device
        self.model = SentimentCheck().to(self.device)
    
        # Hyperparameters
        self.epochs = 15
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        self.total_steps = len(self.train_loader) * self.epochs
        self.scheduler = lr_scheduler.LinearLR(self.optimizer, total_iters=self.total_steps) # reduce lr gradually to 0
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
               
        #----------------TRAIN & VALIDATE MODEL------------------#
        record = defaultdict(list)
        best_val_acc = 0
        for epoch in tqdm(range(self.epochs)):
            print()
            print('=' * 10, f'Epoch {epoch + 1}/{self.epochs}', '=' * 10)
            print()
            
            #----------------TRAINING------------------#
            self.model.train()
            losses = []
            correct_predictions = 0
            
            for batch in self.train_loader:
                # Get inputs for model
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                # Predict and calculate loss
                outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
                _, preds = torch.max(outputs, dim=1)
                loss = self.loss_fn(outputs, labels)
                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
                # Backprop
                loss.backward()
                # Gradient clipping
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Take next step
                self.optimizer.step()
                self.scheduler.step()
                
            train_acc, train_loss = correct_predictions.double() / len(self.trainfile), np.mean(losses)        
            print(f'Train loss {train_loss} - Training accuracy {train_acc}')
            print()
            
            #----------------VALIDATION------------------#
            if self.val_loader is not None:
                self.model = self.model.eval()    
                losses = []
                correct_predictions = 0
                
                with torch.no_grad():
                    for batch in self.val_loader:
                        # Get inputs for model
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        labels = batch["labels"].to(self.device)
                        # Predict and calculate loss
                        outputs = self.model(input_ids = input_ids, attention_mask = attention_mask)
                        _, preds = torch.max(outputs, dim=1)
                        loss = self.loss_fn(outputs, labels)
                        correct_predictions += torch.sum(preds == labels)
                        losses.append(loss.item())
                        
                val_acc, val_loss = correct_predictions.double() / len(self.dev_file), np.mean(losses)
                print(f'Val loss {val_loss} - Val accuracy {val_acc}')
                print()
                
            #----------------RECORD ACCURACY & SAVE BEST MODEL------------------#
            record['train_acc'].append(train_acc)
            record['train_loss'].append(train_loss)
                
            if self.val_loader is not None:
                record['val_acc'].append(val_acc)
                record['val_loss'].append(val_loss)
                
            if (self.val_loader is not None) and (val_acc > best_val_acc):
                torch.save(self.model.state_dict(), 'best_model.pth')
                best_val_acc = val_acc
        
    
    ###############################################################################
    def predict(self, data_filename: str, device: torch.device):
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """
        
        #----------------LOAD FILE & PREPROCESS----------------#
        data_file = pd.read_csv(data_filename, sep='\t', header=None)
        self.data_file = preprocessing(data_file)
        self.dataset = create_dataset(
            sentences=self.data_file[4].to_numpy(),
            aspect_categories = self.data_file[1].to_numpy(),
            target_terms=self.data_file[2].to_numpy(),
            labels = self.data_file[0].to_numpy(),
            tokenizer = self.tokenizer,
            max_len_token = self.max_length)
        self.data_loader = DataLoader(self.dataset, batch_size= self.bs, shuffle=False)
        
        #----------------LOAD MODEL------------------#
        self.device = device
        self.model = SentimentCheck().to(self.device)
        self.model.load_state_dict(torch.load('best_model.pth'))

        #----------------PREDICTING------------------#
        self.model.eval()
        output_labels = []
        label_decode = {0:"negative",1:"neutral",2:"positive"}
        for batch in self.data_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
            logits = F.softmax(outputs, dim=1)
            logits = logits.detach().cpu().numpy()
            outputs = np.argmax(logits, axis=1) # label in number 0,1,2
            for label in outputs:
                output_labels.append(label_decode[label]) # label in text "negative","neutral","positive"
        return np.array(output_labels)
                
                
                
        