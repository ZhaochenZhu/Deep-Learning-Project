import numpy as np
from neuralnet import Neuralnetwork
import copy

def check_grad(model, x_train, y_train):

    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    #Create Model
    #model=Neuralnetwork(config)

    #Defien epsilon and wieght to be tested
    epsilon = 1.e-2
    layer_idx, w_idx_i, w_idx_j=0,999,102


    #Approximation
    #PREP: For a given trained model with layes, each layer has recorded weights, 
    #      (model should return or store a list of layers so I can access and change a single specific weight)
    #TODO:     Use given trian X, forward, til get output, with wij+epsilon, get cross entropy
    #TODO:     Use given trian X, forward, til get output, with wij-epsilon, get cross entropy
    #TODO:     Find diff and approximate gradient
    #copy model:
    m1 = copy.deepcopy(model)
    m2 = copy.deepcopy(model)
    #change single weight (inplace) at given layer, between from ith node in layer L to ith node to layer L+1
    m1.layers[layer_idx].w[w_idx_i][w_idx_j]=m1.layers[layer_idx].w[w_idx_i][w_idx_j]+epsilon
    m2.layers[layer_idx].w[w_idx_i][w_idx_j]=m2.layers[layer_idx].w[w_idx_i][w_idx_j]-epsilon
    #calculate entropy:
    Loss1,acc=m1.forward(x_train,targets=y_train)
    Loss2,acc=m2.forward(x_train,targets=y_train)
    approximate_grad=(Loss1-Loss2)/(2*epsilon)
    # Gradient by Model:
    model.forward(x_train,targets=y_train)
    gradient=-model.backward_check([],gradReqd=True)[::-1][layer_idx][w_idx_i][w_idx_j]
    diff=approximate_grad-gradient
    update_correct=abs(diff)<=(epsilon)**2
    print(layer_idx, w_idx_i, w_idx_j)
    print(approximate_grad)
    print(gradient)
    print(diff)


def checkGradient(x_train,y_train,config):

    subsetSize = 1000  #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    check_grad(model, x_train_sample, y_train_sample)
