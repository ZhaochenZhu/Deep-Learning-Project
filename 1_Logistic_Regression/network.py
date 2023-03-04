import numpy as np
import data
import time


"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    return 1/(1+np.exp(-a))

def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    summation = np.sum(np.exp(a),axis = 1)
    batch_size = len(summation)
    summation = summation.reshape(batch_size,1)
    # print(summation.shape)
    # print(a.shape)
    return np.exp(a)/summation

def binary_cross_entropy(y, t,batch = True):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    #assume y and x are single numbers, not array
    if batch:
        loss =0
        for i in range(len(y)):
            loss+=binary_cross_entropy(y[i],t[i],batch = False)
        return loss
    else:
        return -(t*np.log(y) + (1-t)*np.log((1-y)))

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    return -np.sum(t*np.log(y))

class Network:
    def __init__(self, hyperparameters, activation, loss_func, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.learning_rate = hyperparameters['lr']
        self.activation = activation
        self.loss_func = loss_func

        self.weights = np.zeros((32*32+1, out_dim))
        self.out_dim = out_dim

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        return self.activation(X @ (self.weights))

    def __call__(self, X):
        return self.forward(X)

    def train(self, minibatch):
        # binary=1 --> binary classification_(0,5)
        # binary=0 --> multi-class classification


        # binary=-1 --> binary classification_(3,5)

        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        X, y = minibatch
        X = data.append_bias(X)
        batch_size = len(X)
        if self.out_dim ==1 :
            prediction = self.forward(X).transpose()[0]
            dif = y - prediction # (64,)
            gradient = (dif @ X/batch_size)
            self.weights = self.weights + np.array([self.learning_rate * gradient]).T
            loss = self.loss_func(prediction, y)/batch_size

        else:
            y = data.onehot_encode(y) #different from logistic
            prediction = self.forward(X)
            dif = y - prediction # (64,)
            gradient = -1*(X.T @ dif/batch_size) #different from logistic
            self.weights = self.weights - self.learning_rate * gradient
            loss = self.loss_func(prediction, y)/batch_size
            loss = loss/10
        
        return loss
        



    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """

        X, y = minibatch
        batch_size = len(X) 
        X = data.append_bias(X) # append 1
        if self.out_dim ==1: #if logistic
            # here y is target 
            prediction = self.forward(X).transpose()[0] # make prediction
            loss = self.loss_func(prediction,y)/batch_size # calculate test loss
            prediction = prediction >=0.5 # predict True 1f probability >=0.5 else False
        else:
            #One hot encode y
            encoded_y = data.onehot_encode(y)
            prediction = self.forward(X) #make prediction 
            loss = self.loss_func(prediction,encoded_y)/batch_size #calculate test loss
            loss = loss/10
            prediction = np.argmax(prediction, axis =1) # predict class with highest probability among all classes

        #calculate accuracy
        dif_arr = prediction == y 
        accuracy = np.mean(dif_arr)
        return loss,accuracy
        #return {'loss':loss,'accuracy':accuracy,"prediction":prediction}
        # else:
        #     loss = multiclass_cross_entropy(prediction,y)
        #     label = y
        #     perdiction_label = np.argmax(prediction,axis=1)
        #     diff_arr = label == perdiction_label
        #     accuracy = np.mean(diff_arr)
        # return loss,accuracy


