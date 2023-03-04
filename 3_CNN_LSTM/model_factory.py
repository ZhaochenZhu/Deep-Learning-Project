################################################################################
# CSE 151B: Programming Assignment 3
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin, Yash, James
# Fall 2022
################################################################################
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F
import copy


class CustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(CustomCNN, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.pool21 = nn.MaxPool2d(kernel_size = 3,stride= 2)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features=128,out_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=1024)
        self.fc3 = nn.Linear(in_features=1024,out_features=outputs)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.batch_norm2 = nn.BatchNorm2d(num_features=128)
        self.batch_norm3 = nn.BatchNorm2d(num_features=256)
        self.batch_norm4 = nn.BatchNorm2d(num_features=256)
        self.batch_norm5 = nn.BatchNorm2d(num_features=128)

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        # -> n, 3, 258, 258
        x= x.to(self.device)
        x = self.pool1(F.relu(self.batch_norm1(self.conv1(x)))) 
        x = self.pool2(F.relu(self.batch_norm2(self.conv2(x)))) 
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool21(F.relu(self.batch_norm5(self.conv5(x))))
        x = self.pool3(x)
        x = torch.flatten(x,start_dim =1) #squeeze()?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN_LSTM(nn.Module):
    '''
    An encoder decoder architecture.
    Contains a reference to the CNN encoder based on model_type config value.
    Contains an LSTM implemented using PyTorch modules. This will serve as the decoder for our Image Captioning problem.
    '''
    def __init__(self, config_data, vocab):
        '''
        Initialize the embedding layer, LSTM, and anything else you might need.
        '''
        super(CNN_LSTM, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.hidden_size = config_data['model']['hidden_size']
        self.embedding_size = config_data['model']['embedding_size']
        self.model_type = config_data['model']['model_type']
        self.max_length = config_data['generation']['max_length']
        self.deterministic = config_data['generation']['deterministic']
        self.temp = config_data['generation']['temperature']
        self.batch_size = config_data['dataset']['batch_size']
        self.device = torch.device('cuda')
        # TODO
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embedding_size)
        self.model = nn.LSTM(input_size = self.embedding_size,num_layers = 2, hidden_size = self.hidden_size, batch_first = True)
        self.outlayer = nn.Linear(in_features=self.hidden_size,out_features=self.vocab_size)
        self.softmax = nn.Softmax(dim=2)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")


    def forward(self, images, captions,batch_size, teacher_forcing=False):
        '''
        Forward function for this model.
        If teacher forcing is true:
            - Pass encoded images to the LSTM at first time step.
            - Pass each encoded caption one word at a time to the LSTM at every time step after the first one.
        Else:
            - Pass encoded images to the LSTM at first time step.
            - Pass output from previous time step through the LSTM at subsequent time steps
            - Generate predicted caption from the output based on whether we are generating them deterministically or not.
        '''
        h0 = torch.zeros(2,batch_size,self.hidden_size,device=self.device) #dont initialize hidden state
        c0 = torch.zeros(2,batch_size,self.hidden_size,device=self.device)
        images = images.reshape(batch_size, 1, self.embedding_size) #batch_size, 1, 300
        images = images.to(self.device)
        if not captions is None:
            captions = captions.to(self.device)
        if teacher_forcing:
            captions_embedding = self.embedding(captions)          
            input = torch.cat((images,captions_embedding),dim=1)# bs, L, 300
            output,hidden_state = self.model(input)   #bs, L, 512                
            output = self.outlayer(output) #bs L 14463
            return output
        else:
            seq = None
            seq_len = 0
            index,h_temp = self.model(images,(h0,c0))
            #index: Batch_size,1,512
            while seq_len < self.max_length:
                index = self.outlayer(index) #Batch_size, 1, 14463
                index = self.softmax(index/self.temp)#Batch_size, 1, 14463
                if self.deterministic:
                    index = torch.argmax(index,dim=2)#Batch_size, 1
                else:
                    index = torch.multinomial(input = index.squeeze(1),num_samples= 1)
#                 if seq_len == 0:
#                     print(index)
                if seq is None:
                    seq = copy.deepcopy(index)
                else:
                    seq = torch.cat((seq, index),dim=1)
                seq_len+=1
                index = self.embedding(index)
                index,h_temp = self.model(index,h_temp)
            return seq
                

#             #start_signal here is unsure of whether it is <start>, come back later and check

# This class are for CNN architecture tunning purpose
#This Class stands for Deep CustomCNN_LSTM model
class DeepCustomCNN(nn.Module):
    '''
    A Custom CNN (Task 1) implemented using PyTorch modules based on the architecture in the PA writeup. 
    This will serve as the encoder for our Image Captioning problem.
    '''
    def __init__(self, outputs):
        '''
        Define the layers (convolutional, batchnorm, maxpool, fully connected, etc.)
        with the correct arguments
        
        Parameters:
            outputs => the number of output classes that the final fully connected layer
                       should map its input to
        '''
        super(DeepCustomCNN, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=11,stride=4)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=5,padding=2)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1) #changed
        self.conv6 = nn.Conv2d(in_channels=512, out_channels = 512,kernel_size = 2, padding = 1)#added
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.pool21 = nn.MaxPool2d(kernel_size = 3,stride= 2)
        self.pool3 = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features=512,out_features=1024)
        self.fc2 = nn.Linear(in_features=1024,out_features=2048)
        self.fc3 = nn.Linear(in_features=2048,out_features=outputs)
        self.batch_norm1 = nn.BatchNorm2d(num_features=64)
        self.batch_norm2 = nn.BatchNorm2d(num_features=128)
        self.batch_norm3 = nn.BatchNorm2d(num_features=256)
        self.batch_norm4 = nn.BatchNorm2d(num_features=256)
        self.batch_norm5 = nn.BatchNorm2d(num_features=512)
        self.batch_norm6 = nn.BatchNorm2d(num_features=512)#added

    def forward(self, x):
        '''
        Pass the input through each layer defined in the __init__() function
        in order.

        Parameters:
            x => Input to the CNN
        '''
        # -> n, 3, 258, 258
        x= x.to(self.device)
        x = self.pool1(F.relu(self.batch_norm1(self.conv1(x)))) #conv1 -> n , 64,62,62 ->pool1 n,64,30,30
        x = self.pool2(F.relu(self.batch_norm2(self.conv2(x)))) #conv2 => n,128, 30,30 ->pool2 n, 128,14,14
        x = F.relu(self.batch_norm3(self.conv3(x)))#conv3 -> n,256,14,14 
        x = F.relu(self.batch_norm4(self.conv4(x)))#conv4 ->n,256,14,14
        x = self.pool21(F.relu(self.batch_norm5(self.conv5(x))))#conv5 ->n,512,14,14 ->pool21 n,512,6,6
        x= F.relu(self.batch_norm6(self.conv6(x))) #conv6 ->n,512,7,7
        x = self.pool3(x) #->pool3 n,512,1,1
        x = torch.flatten(x,start_dim =1) #squeeze()?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet(nn.Module):
    def __init__(self, embedding_size):
        super(ResNet,self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        self.model = resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        # self.fc = nn.Linear(self.model.fc.in_features,embedding_size)
        # self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.fc = nn.Linear(self.model.fc.in_features,embedding_size)
        

    def forward(self,x):
        x= x.to(self.device)
        x = self.model(x)
        # x = self.fc(x)
        return x

class completeModel(nn.Module):
    def __init__(self, config_data,vocab):
        super(completeModel,self).__init__()
        if(config_data["model"]['model_type']=="Custom"):
            self.encoder = CustomCNN(config_data['model']['embedding_size'])
        elif (config_data["model"]['model_type']=="DeepCustom"):
            self.encoder = DeepCustomCNN(config_data['model']['embedding_size'])
        else:
            self.encoder = ResNet(config_data['model']["embedding_size"])
        self.decoder = CNN_LSTM(config_data=config_data,vocab=vocab)
    def forward(self, x, captions, batch_size,teacher_forcing=False):
        x = self.encoder.forward(x) #bs, 300
#         x = self.fc(x)
        output = self.decoder.forward(x,captions,batch_size,teacher_forcing)
        return output
        

def get_model(config_data, vocab):
    '''
    Return the LSTM model
    '''
    model = completeModel(config_data,vocab)
    return model
