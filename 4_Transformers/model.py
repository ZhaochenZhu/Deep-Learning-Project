import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig
from transformers import AdamW, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

class IntentModel(nn.Module):
  def __init__(self, args, tokenizer, target_size):
    super().__init__()
    self.tokenizer = tokenizer
    self.model_setup(args)
    self.target_size = target_size

    # task1: add necessary class variables as you wish.
    self.optimizer = None
    self.scheduler = None
    self.mode=args.task
    # task2: initilize the dropout and classify layers
    self.dropout = nn.Dropout(p=args.drop_rate)
    self.classify = Classifier(args=args,target_size=self.target_size)
    
  def model_setup(self, args):
    print(f"Setting up {args.model} model")

    # task1: get a pretrained model of 'bert-base-uncased'
    self.encoder = BertModel.from_pretrained("bert-base-uncased")
    
    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check

  def forward(self, inputs, targets):
    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the output of the dropout layer to the Classifier which is provided for you.
    """
    text = inputs['input_ids']
    # encoded_input = self.tokenizer(text)
    # print(len(text[0]))
    embedded = self.encoder(text) 
    # print(embedded[0].shape)
    # print(embedded[1].shape)
    if self.mode == 'baselinedraw': #AAA
        #print(self.mode)
        return embedded[0][:,0,:] #AAA
    dropout = self.dropout(embedded[0][:,0,:])# should we use the first element of hidden?
    
    # print(dropout.shape)
    output = self.classify(dropout)
    return output
    
    """
    By default output = BertModel.from_pretrained('bert-base-uncased') is a 2-tuple where output[0] 
    is the hidden states of the last layer, but how is output[1] computed?
    "Last layer hidden-state of the first token of the sequence (classification token) further processed 
    by a Linear layer and a Tanh activation function. The Linear layer weights are trained from the next 
    sentence prediction (classification) objective during Bert pretraining. This output is usually not a 
    good summary of the semantic content of the input, you’re often better with averaging or pooling the 
    sequence of hidden-states for the whole input sequence."
    """
    
  
class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim #embede_dim = 10
    self.top = nn.Linear(input_dim, args.hidden_dim) #hidden_dim =10
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size) #target size = 60
    

  def forward(self, hidden):
    middle = self.relu(self.top(hidden))
    logit = self.bottom(middle)
    return logit


class CustomModel(IntentModel):
  def __init__(self, args, tokenizer, target_size):
    super().__init__(args, tokenizer, target_size)
    input_dim = args.embed_dim #embede_dim = 10
    self.top = nn.Linear(input_dim, args.hidden_dim) #hidden_dim =10
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size) #target size = 60
    self.n_init_layer = args.reinit_n_layers
    print(self.n_init_layer)
    for n in range(self.n_init_layer):
        self.encoder.encoder.layer[-(n+1)].apply(self.encoder._init_weights)

class SupConModel(IntentModel):
  def __init__(self, args, tokenizer, target_size, feat_dim=768):
    super().__init__(args, tokenizer, target_size)

    # task1: initialize a linear head layer
    input_dim = args.embed_dim
    self.head = nn.Linear(in_features=input_dim, out_features=feat_dim)
    self.dropout = nn.Dropout(p=args.drop_rate)
    self.encoder = BertModel.from_pretrained("bert-base-uncased")
    # self.normalize = nn.functional.normalize()
    self.mode=args.task
 
  def forward(self, inputs, targets):

    """
    task1: 
        feeding the input to the encoder, 
    task2: 
        take the last_hidden_state's <CLS> token as output of the
        encoder, feed it to a drop_out layer with the preset dropout rate in the argparse argument, 
    task3:
        feed the normalized output of the dropout layer to the linear head layer; return the embedding
    """
    text = inputs['input_ids']
    embedded = self.encoder(text) 
    #print(output.shape)
    if self.mode == 'supcondraw':#AAA
        #print(self.mode)   
        return embedded[0][:,0,:]#AAA
    dropout = self.dropout(embedded[0][:,0,:])
    normalized = F.normalize(dropout,dim=1) # batchsize, sequence_length, 768
    output = self.head(normalized)
    return output
    
