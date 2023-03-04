import os, sys, pdb
import numpy as np
import random
import torch

import math

from tqdm import tqdm as progress_bar

from utils import set_seed, setup_gpus, check_directories
from dataloader import get_dataloader, check_cache, prepare_features, process_data, prepare_inputs
from load import load_data, load_tokenizer
from arguments import params
from model import IntentModel, SupConModel, CustomModel
from torch import nn
import torch.optim as optim

import transformers

if torch.cuda.is_available():
  device='cuda'
else:
  device = "cpu"

#for plotting
import matplotlib.pyplot as plt
import umap
  
  
  
def baseline_train(args, model, datasets, tokenizer):
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args=args, dataset=datasets["train"])
    print(train_dataloader)

    # task2: setup model's optimizer_scheduler if you have
    model.optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    model.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=model.optimizer,T_max=args.n_epochs)
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in enumerate(progress_bar(train_dataloader)):
            inputs, labels = prepare_inputs(batch=batch,model=model)
            logits = model(inputs,labels)
            loss = criterion(logits,labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            model.scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()/len(train_dataloader)
    
        run_eval(args=args, model=model,datasets=datasets, tokenizer=tokenizer, split='validation')
        print('epoch', epoch_count+1, '| losses:', losses/args.n_epochs)

        #save model as pt
        best_model = "best_model_baseline.pt" #AAA
        model_dict = model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': model.optimizer.state_dict()}
        torch.save(state_dict, best_model) #AAA
  
def custom_train(args, model, datasets, tokenizer):
    print("custom_train")
    criterion = nn.CrossEntropyLoss()  # combines LogSoftmax() and NLLLoss()
    # task1: setup train dataloader
    train_dataloader = get_dataloader(args=args, dataset=datasets["train"])

    # task2: setup model's optimizer_scheduler if you have
    model.optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    model.scheduler = transformers.get_linear_schedule_with_warmup(optimizer = model.optimizer,num_warmup_steps = 5,num_training_steps = args.n_epochs)
#     model.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=model.optimizer,T_max=args.n_epochs)
    
#     swa_model = optim.swa_utils.AveragedModel(model)
#     swa_scheduler = optim.swa_utils.SWALR(model.optimizer, swa_lr=0.05)
#     swa_start = 4
    
    # task3: write a training loop
    for epoch_count in range(args.n_epochs):
        losses = 0
        model.train()

        for step, batch in enumerate(progress_bar(train_dataloader)):
            inputs, labels = prepare_inputs(batch=batch,model=model)
            logits = model(inputs,labels)
            loss = criterion(logits,labels)
            loss.backward()

            model.optimizer.step()  # backprop to update the weights
            # Update learning rate schedule
            model.zero_grad()
            losses += loss.item()/len(train_dataloader)
           
        model.scheduler.step() 
            
        run_eval(args=args, model=model,datasets=datasets, tokenizer=tokenizer, split='validation')
        print('epoch', epoch_count, '| losses:', losses/args.n_epochs)

def run_eval(args, model, datasets, tokenizer, split='validation'):
    model.eval()
    dataloader = get_dataloader(args, datasets[split], split)

    acc = 0
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        # print(batch)
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)
        
        tem = (logits.argmax(1) == labels).float().sum()
        acc += tem.item()
  
    print(f'{split} acc:', acc/len(datasets[split]), f'|dataset split {split} size:', len(datasets[split]))
def supcon_train(args, model, datasets, tokenizer):
    from loss import SupConLoss
    criterion = SupConLoss(temperature=args.temperature)

    # task1: load training split of the dataset
    train_dataloader = get_dataloader(args=args, dataset=datasets["train"])
    print(train_dataloader)
    
    # task2: setup optimizer_scheduler in your model
    #model.optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    model.optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=model.optimizer,T_max=args.n_epochs)    

    # task3: write a training loop for SupConLoss function 
    for epoch_count in range(args.n_epochs):
      losses = 0
      model.train()
      
      for step, batch in enumerate(progress_bar(train_dataloader)):
        inputs, labels = prepare_inputs(batch=batch,model=model)
        
        if(args.supconloss=='true'):
          logit1 = model(inputs,labels) #bs*feature dim of supercon
          logit2 = model(inputs,labels)
          logits = torch.cat([logit1.unsqueeze(1), logit2.unsqueeze(1)], dim=1)
#           print(logit1.unsqueeze(1).shape)
#           print(labels.shape)
#           print(logits.shape)
          loss = criterion(logits,labels) # bs 
        else:
          #To use SimCSE loss (unsupervised)
          #print('SimCSE')
          logit1 = model(inputs,labels)
          logit2 = model(inputs,labels)
          logits = torch.cat([logit1.unsqueeze(1), logit2.unsqueeze(1)], dim=1)
          loss = criterion(logits)
        loss.backward()
        
        model.optimizer.step()  # backprop to update the weights
        
        model.zero_grad()
        
        losses += loss.item()/len(train_dataloader)
      model.scheduler.step()  # Update learning rate schedule
      print('epoch', epoch_count+1, '| losses:', losses/args.n_epochs)
        
    #To Save the Model
    if(args.supconloss=='true'):  
      best_model = "best_model_supcon.pt" #AAA
    else:
      best_model = "best_model_simCSE.pt" #AAA
    
    model_dict = model.state_dict()
    state_dict = {'model': model_dict, 'optimizer': model.optimizer.state_dict()}
    torch.save(state_dict, best_model) #AAA
    
def model_draw(args, model, datasets, tokenizer, best_model_file = None):
    '''To Read the stored Model and Draw the Embedding CLS in 2D spaceusing using UMAP '''
    model_status='initial'
    #if given path ot model pt: load the best model
    if best_model_file is not None:
        best_model = best_model_file
        params = torch.load(best_model,map_location = device)['model']
        model.load_state_dict(params) #load best model train (if comment out: use a initail random model)
        model_status='trained'
        print('load trained model from', best_model_file)
    model.to(device)
    split='test'
    dataloader = get_dataloader(args, datasets[split], split)
    print('load test data')
    
    #plot
    fig, ax=plt.subplots()
    
    #RUN ON TEST SET
    #embeddings=torch.empty(1, 768)
    embeddings=np.random.rand(1, 768)
    label_10=np.random.rand(1)
    model.eval()
    for step, batch in progress_bar(enumerate(dataloader), total=len(dataloader)):
        # print(batch)
        inputs, labels = prepare_inputs(batch, model)
        logits = model(inputs, labels)[labels<10].cpu().detach().numpy()

        #embeddings=torch.cat([embeddings, logits], dim=0)
        embeddings=np.append(embeddings,logits,axis=0)
        label_10=np.append(label_10,labels[labels<10].cpu().detach().numpy(),axis=0)
    #use UMAP to embed
    print('embedding')
    embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.01,
                      metric='correlation').fit_transform(embeddings[1:]) #[1:] to cut the initial random tensor
    print('embedded')
    
    #plot the 2d features
    color_pack=['r','g','y','b','c','m','coral','grey','pink','purple']
    colors=[color_pack[int(i)] for i in label_10[1:]]
    ax.scatter(embeddings[1:,0],embeddings[1:,1],c=colors, alpha=0.7)
    print('draw, save plot')
        
    fig.savefig(f"output_{model_status}_{args.task}.jpg")
    print(f'save as output_{model_status}_{args.task}.jpg')
        
        
if __name__ == "__main__":
  args = params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)

  if already_exist:
    features = cache_results
  else:
    data = load_data()
    features = prepare_features(args, data, tokenizer, cache_results)
  datasets = process_data(args, features, tokenizer)
  for k,v in datasets.items():
    print(k, len(v))
 
  if args.task == 'baseline':
    print(device)
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    baseline_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'custom': # you can have multiple custom task for different techniques
    model = CustomModel(args, tokenizer, target_size=60).to(device)
    run_eval(args, model, datasets, tokenizer, split='validation')
    run_eval(args, model, datasets, tokenizer, split='test')
    custom_train(args, model, datasets, tokenizer)
    run_eval(args, model, datasets, tokenizer, split='test')
  elif args.task == 'supcon':
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    supcon_train(args, model, datasets, tokenizer)

  #for drawing output
  elif args.task == 'supcondraw': #AAA
    if(args.supconloss=='true'):  
      best_model = "best_model_supcon.pt" #AAA
    else:
      best_model = "best_model_simCSE.pt" #AAA
    print('draw for ' , best_model)
    model = SupConModel(args, tokenizer, target_size=60).to(device)
    model_draw(args, model, datasets, tokenizer, best_model_file=best_model)
  elif args.task == 'baselinedraw': #AAA
    model = IntentModel(args, tokenizer, target_size=60).to(device)
    model_draw(args, model, datasets, tokenizer, best_model_file="best_model_baseline.pt")
   

   
