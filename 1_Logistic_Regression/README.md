# Trained Network to do Image Classification with Logistic Regression and Softmax Regression
In this project, we've train our model using the data from cifar-10, we've tested different hyperparameters(batch_size, learning_rate and normalization function). When running the program, the output will tell you the best loss we get in each of the 10 cross validation process, we used this to compute the average loss, which indicate the performance of the current model. The best models we got are given below(hyperparameters fine tuned).

## Package used
### numpy
We use numpy to do do matrix operation and make our code more simplified and readable
### os/pickle
We use os and pickle to get data from cifar-10 dataset
### time
We use time to keep track of the runtime of each epoch and the whole program
### matplotlib
We use matplotlib.pyplot to visualize our training performance

## Methodology
### Logistic Regression: 
Logistic regression is a binary classification method. Intuitively, logistic regression can be conceptualized as a single neuron reading in a d-dimensional input vector x ∈ Rd and producing an output y between 0 and 1 that is the system’s estimate of the conditional probability that the input is in the target category, given the input. (Taken from the write-up)
### Softmax Regression:
Softmax regression is the generalization of logistic regression for multiple (c) classes. Now given an input x<sup>n</sup>, softmax regression will output a vector y<sup>n</sup>, where each element, y<sup>n</sup><sub>k</sub> represents the probability that x<sup>n</sup> is in class k. (Taken from the write-up)

## How to run the code:
### The best training model for binary classification between set 0 and 5
Run the following command:  
python -m main --batch_size 4 --epochs 100 --learning_rate 0.001 --z_score min --binary 1  
The average accuracy on validation set will be around 0.723 
The accuracy on test set will be around 0.718     

## The best training model for binary classification between set 3 and 5
Run the following command:  
python -m main --batch_size 10 --epochs 100 --learning_rate 0.0005 --z_score min --binary -1  
The average accuracy on validation set will be around 0.587   
The accuracy on test set will be around 0.594     

## The best training model for multi-class classification
Run the following command:  
python -m main --batch_size 128 --epochs 100 --learning_rate 0.005 --z_score min --binary 0  
The average accuracy on validation set will be around 0.299  
The accuracy on test set will be around 0.303     