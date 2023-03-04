################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################

import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
import math
from tqdm import tqdm
from copy import deepcopy
import nltk
from nltk.tokenize import word_tokenize
import caption_utils
ROOT_STATS_DIR = './experiment_data'
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import copy


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        self.config_data = read_file_in_dir('./', name + '.json')
        if self.config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = self.config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco, self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            self.config_data)
        self.vocab = self.__vocab.idx2word
        self.vocab_size = len(self.vocab)

        # Setup Experiment
        self.__epochs = self.config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__early_stop = self.config_data['experiment']['early_stop']
        self.__patience = self.config_data['experiment']['patience']
        self.__batch_size = self.config_data['dataset']['batch_size']

        # Init Model
        self.__model = get_model(self.config_data, self.__vocab)
        self.__best_model = None

        # criterion
        self.__criterion = nn.CrossEntropyLoss()  # TODO

        # optimizer
        self.__optimizer = optim.Adam(self.__model.parameters(),lr=self.config_data['experiment']['learning_rate'])

        # LR Scheduler
        self.__LR_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.__optimizer,T_max=self.__epochs)

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

        self.device =  torch.device('cuda:0')
        # raise NotImplementedError()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

#             state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model.'))
            state_dict = torch.load("best_model.pt",map_location = torch.device("cuda:0"))
            self.__model.load_state_dict(state_dict['model'])
#             self.__optimizer.load_state_dict(state_dict)
        else:
            os.makedirs(self.__experiment_dir,exist_ok=True)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        print("start running")
        start_epoch = self.__current_epoch
        patience_count = 0
        min_loss = 100
        train_losses = []
        print(start_epoch)
        print(self.__epochs)
        for epoch in range(start_epoch,self.__epochs):  # loop over the dataset multiple times
            print(f'Epoch {epoch + 1}')
            print('--------')
            start_time = datetime.now()
            self.__current_epoch = epoch
            print('Training...')
            print('-----------')
            train_loss = self.__train()
            train_losses.append(train_loss)
            print('Validating...')
            print('-------------')
            val_loss = self.__val()

            # save best model
            if val_loss < min_loss:
                min_loss = val_loss
                self.__best_model = "best_model.pt"
                model_dict = self.__model.state_dict()
                state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
                torch.save(state_dict, self.__best_model)

            # early stop if model starts overfitting
            if self.__early_stop:
                if epoch > 0 and val_loss > self.__val_losses[epoch - 1]:
                    patience_count += 1
                if patience_count >= self.__patience:
                    print('\nEarly stopping!')
                    self.__record_stats(train_loss, val_loss)
                    break

            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if self.__LR_scheduler is not None:
                self.__LR_scheduler.step()
        if not self.__best_model is None:
            params = torch.load(self.__best_model,map_location = self.device)['model']
            self.__model.load_state_dict(params)

    def __compute_loss(self, images, captions):
        """
        Computes the loss after a forward pass through the model
        """
        loss = self.__criterion(images,captions)
        

    def __train(self):
        """
        Trains the model for one epoch using teacher forcing and minibatch stochastic gradient descent
        """
        total_loss = 0
        for i in enumerate(tqdm(self.__train_loader)):
            self.__optimizer.zero_grad()
            train_image = i[1][0] #batchsize ,3, 256,256
            train_image.to(self.device)
            train_captions = i[1][1]
            train_captions.to(self.device)            
            batch_size = train_image.shape[0]
            output = self.__model.forward(x=train_image,captions=copy.deepcopy(train_captions[:,:-1]),batch_size=batch_size,teacher_forcing=True)
            sen_len =output.shape[1]
            output = output.reshape(batch_size*sen_len,self.vocab_size)
            train_captions = train_captions.reshape(batch_size*sen_len).to(self.device)
            loss = self.__criterion(output,train_captions)
            total_loss += loss.item()
            loss.backward()
            self.__optimizer.step()
        return total_loss/len(self.__train_loader)

    def __generate_captions(self,img_id,sent,testing):
        """
        Generate captions without teacher forcing
        Params:
            img_id: Image Id for which caption is being generated
            outputs: output from the forward pass for this img_id
            testing: whether the image_id comes from the validation or test set
        Returns:
            tuple (list of original captions, predicted caption)
        """
        prediction = []
        for ind in sent:
            word = self.vocab[ind]
            prediction.append(self.vocab[ind])
        if testing:
            origins = []
            captions = self.__coco_test.imgToAnns[img_id]
            for i in captions:
                origin = i['caption']
                origin = nltk.word_tokenize(origin)
                origins.append(origin) #remember to change it back to coco_test
        else:
            origins = []
            captions = self.__coco.imgToAnns[img_id]
            for i in captions:
                origin = i['caption']
                origin = nltk.word_tokenize(origin)
                origins.append(origin)
        print(prediction)
        return origins, prediction

    def __str_captions(self, img_id, original_captions, predicted_caption):
        """
            !OPTIONAL UTILITY FUNCTION!
            Create a string for logging ground truth and predicted captions for given img_id
        """
        result_str = "Captions: Img ID: {},\nActual: {},\nPredicted: {}\n".format(
            img_id, original_captions, predicted_caption)
        return result_str

    def __val(self):
        """
        Validate the model for one epoch using teacher forcing
        """
        total_loss = 0
        for i in enumerate(tqdm(self.__val_loader)):
            val_image = i[1][0]
            val_image.to(self.device)
            # val_captions = F.one_hot(i[1][1],num_classes=self.vocab_size)
            val_captions = i[1][1]
            val_captions.to(self.device)
            batch_size = val_image.shape[0]
            self.__optimizer.zero_grad()
            output = self.__model.forward(x=val_image,captions=copy.deepcopy(val_captions[:,:-1]),batch_size= batch_size,teacher_forcing=True)
            sen_len =output.shape[1]
            output = output.reshape(batch_size*sen_len,self.vocab_size)
#             val_captions =  F.one_hot(train_captions,num_classes = self.vocab_size)
            val_captions = val_captions.reshape(batch_size*sen_len).to(self.device)
            loss = self.__criterion(output,val_captions)
            total_loss += loss.item()
            # loss.backward()
            # self.__optimizer.step()
            # self.__LR_scheduler.step()            
        return total_loss/len(self.__val_loader)

    def test(self):
        """
        Test the best model on test data. Generate captions and calculate bleu scores
        """
        total_loss = 0
        bleu1 = 0
        bleu4 = 0
        if not self.__best_model is None:
            self.__model =  get_model(self.config_data, self.__vocab)
            params = torch.load(self.__best_model,map_location = self.device)['model']
            self.__model.load_state_dict(params)
            self.__model.to(self.device) 
        for i in enumerate(tqdm(self.__test_loader)):
            test_image = i[1][0] #batch_size,3,256,256
            test_image.to(self.device)
            # test_captions = F.one_hot(i[1][1],num_classes=self.vocab_size)
            test_captions = i[1][1]
#             print(f"csefasefsefSSSSS:{test_captions}")
            test_captions.to(self.device)
            test_image_id = i[1][2]
            batch_size = test_image.shape[0]
            seq = self.__model.forward(test_image,copy.deepcopy(test_captions[:,:-1]),batch_size,False)
            output = self.__model.forward(test_image,copy.deepcopy(test_captions[:,:-1]),batch_size,True)
            sen_len =output.shape[1]
            output = output.reshape(batch_size*sen_len,self.vocab_size)
#             test_captions = F.one_hot(test_captions,num_classes=self.vocab_size)
            test_captions = test_captions.reshape(batch_size*sen_len).to(self.device)
            loss = self.__criterion(output,test_captions)
            total_loss+=loss.item()
            seq = seq.cpu().detach().numpy()
            sentence =[]
            bleu1_temp = []
            bleu4_temp = []
            for item in range(len(seq)):
                sent = seq[item]
                print(f"start of sent: {self.vocab[sent[0]]}")
                img_id = test_image_id[item]
                origin, predicted = self.__generate_captions(img_id, sent,testing= True)
                print(f"origin: {origin}")
                print(f"predicted: {predicted}")
                bleu1_temp.append(caption_utils.bleu1(origin,predicted))
                bleu4_temp.append(caption_utils.bleu4(origin,predicted))
            bleu1 = np.mean(bleu1_temp)
            bleu4 = np.mean(bleu4_temp)

        print(f"average test loss is{total_loss}, average bleu1 is {bleu1}, average bleu4 is {bleu4}")

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
