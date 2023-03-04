
# from hamcrest import none
import numpy as np
# from pyrsistent import T
# from sqlalchemy import false, true
import util
import copy

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        TODO: Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU","output"]:   #output can be used for the final layer. Feel free to use/remove it
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        TODO
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        # print(self.activation_type)
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            # print("activation backward tanh")
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        # print("dimension before tanh")
        # print(x.shape)
        return np.tanh(x)

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(0,x)

    def output(self, x):
        """
        TODO: Implement softmax function here.
        Remember to take care of the overflow condition.
        """
        summation = np.sum(np.exp(x),axis = 1)
        batch_size = len(summation)
        summation = summation.reshape(batch_size,1)
        # print(summation.shape)
        # print(a.shape)
        return np.exp(x)/summation

    def grad_sigmoid(self,x):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def grad_tanh(self,x):
        """
        TODO: Compute the gradient for tanh here.
        """
        # print("dimension before g' tanh")
        # print(x.shape)
        return 1-self.tanh(x)**2

    def grad_ReLU(self,x):
        """
        TODO: Compute the gradient for ReLU here.
        """
        res = np.zeros(x.shape)
        res[x>0] = 1
        return res

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """

        return 1


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation, weightType):
        """
        TODO
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        self.w = np.zeros((in_units+1, out_units))
        if (weightType == 'random'):
            self.w = 0.01 * np.random.random((in_units + 1, out_units))

        self.x = None    # Save the input to forward in this
        self.a = None    #output without activation
        self.z = None    # Output After Activation
        self.activation = activation   #Activation function
        self.previous_weight_change = np.zeros((in_units+1, out_units))

        self.dw = 0  # Save the gradient w.r.t w in this. You can have bias in w itself or uncomment the next line and handle it separately
        # self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        TODO
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        self.x = x

        self.a = np.dot(x, self.w)
        self.z = self.activation.forward(self.a)
        return self.z
        

    def backward(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True):
        """
        TODO: Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update the weights i.e. whether self.w should
        be updated after calculating self.dw
        The delta expression (that you prove in PA1 part1) for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass
        """
        # g'(a_i) dimension batch_size * 128, batch_size*3072
        gradient = self.activation.backward(self.a)


        # weight dimension: 129*10 for output
        # deltacur batch_size*outDimension --> 256*10 , #deltacur batch_size*128
        weight_no_bias = self.w[:-1] #128*10 #3072,128
        delta = np.multiply(gradient, deltaCur) #batch_size*128
        
        self.dw = delta.dot(weight_no_bias.T) #batch_size* 128
        # TODO
        # Implement backprop with momentum_gamma and regularization
        regu_term = regularization*self.w #L2
        #regu_term = regularization*(self.w/np.abs(self.w)) #L1
        if gradReqd:
            #self.previous_weight_change 129*10  self.x.T.dot(deltaCur) 129*10
            weight_change = (momentum_gamma*self.previous_weight_change 
                             + 
                             learning_rate*(self.x.T.dot(delta))/len(self.x)
                             - 
                             learning_rate*regu_term
                            )
            #self.previous_weight_change 129*10
            self.previous_weight_change = weight_change
            #self.w 129*10
            self.w = self.w + weight_change
        return self.dw #1*128
    def backward_check(self, deltaCur, learning_rate, momentum_gamma, regularization, gradReqd=True, ):
        """
        Backward for checking gradient
        """
        #calculate slope,delta
        slope = self.activation.backward(self.a)
        #print(slope)
        weight_no_bias = self.w[:-1] #128*10 #3072,128
        delta = np.multiply(slope, deltaCur) #batch_size*128
        regu_term_L2 = regularization*self.w
        #print(delta.shape)
        self.dw = delta.dot(weight_no_bias.T) #batch_size* 128
        gradient=(self.x.T.dot(delta))/len(self.x)#-self.previous_weight_change 129*10
        if gradReqd:
            #self.previous_weight_change 129*10  self.x.T.dot(deltaCur) 129*10
            #weight_change = momentum_gamma*self.previous_weight_change + (1-momentum_gamma)*learning_rate*gradient
            weight_change = momentum_gamma*self.previous_weight_change + learning_rate*gradient-learning_rate*regu_term_L2 #regularization term
            #regularization: 
            self.previous_weight_change = weight_change
            #self.w 129*10
            self.w = self.w + weight_change
        return self.dw, gradient


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        """
        TODO
        Create the Neural Network using config. Feel free to add variables here as per need basis
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1 # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None        # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.lr = config['learning_rate']
        self.regularization = config['L2_penalty']
        self.momentum_gamma = config['momentum_gamma']

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                # print(config['layer_specs'][i])
                # print(config['layer_specs'][i + 1])
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                          config["weight_type"]))
            elif i == self.num_layers - 1:
                # print(config['layer_specs'][i])
                # print(config['layer_specs'][i + 1])
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))

    def __call__(self, x, targets=None):
        """
        TODO
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)


    def forward(self, x, targets=None):
        """
        TODO: Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        """
        # x shape is : 128*3072
        self.x = x
        self.targets = targets
        inputForLayer = x #batch size * 3072
        for layer in self.layers:
            inputForLayer = util.append_bias(inputForLayer) #batch size *3073 #batch_size * 129
            inputForLayer = layer.forward(inputForLayer) #batch size *128 #batch_size * 10
        self.y = inputForLayer #batch_size * 10
        acc = util.calculateCorrect(self.y,targets)
        loss = self.loss(self.y,targets)
        return loss,acc
        
        # if not targets provided, return the output

    def loss(self, logits, targets):
        '''
        TODO: compute the categorical cross-entropy loss and return it.
        '''
        loss = -np.sum(targets*np.log(logits),axis=1)
        for layer in self.layers:
            loss = loss + self.regularization*(np.sum(layer.w ** 2)) #L2
            #loss = loss + self.regularization*(np.sum(np.abs(layer.w))) #L1
        return np.mean(loss)
        

    def backward(self, gradReqd=True):
        '''
        TODO: Implement backpropagation here by calling backward method of Layers class.
        Call backward methods of individual layers.
        '''

        delta = self.targets-self.y #batch_size * 10
        for layer in self.layers[::-1]:
            delta = layer.backward(delta, learning_rate=self.lr, momentum_gamma=self.momentum_gamma,regularization = self.regularization , gradReqd=gradReqd)
            # 1*128
            
    def backward_check(self, gradients,gradReqd=True):
        '''
        check Gradients
        '''

        delta = self.targets-self.y #batch_size * 10
        for layer in self.layers[::-1]:
            delta, gradient = layer.backward_check(delta, learning_rate=self.lr, momentum_gamma=self.momentum_gamma,regularization = self.regularization , gradReqd=gradReqd)
            # 1*128
            gradients.append(gradient)
        return gradients




