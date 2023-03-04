
import copy
from matplotlib import pyplot as plt
from neuralnet import *
from util import *
import time

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Learns the weights (parameters) for our model
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """

    # Read in the esssential configs
    x_train,y_train = shuffle(x_train,y_train)[:25000]
    x_valid,y_valid = shuffle(x_valid,y_valid)[:3000]

    # lr = config['learning_rate']
    batchsize = config['batch_size']
    epochs = config['epochs']
    early_stop = config['early_stop']
    early_stop_epoch = config['early_stop_epoch']
    

    best_val_loss = float('inf')
    val_accs = []
    val_losses = []
    train_accs = []
    train_losses = []
    patient = 0

    for epoch in range (epochs):
        dataset = (x_train,y_train)
        dataset = util.shuffle(dataset[0],dataset[1])
        batches = util.generate_minibatches(dataset,batchsize)
        start = time.time()
        print(f'current epoch is {epoch+1}')
        for batch in batches:
            model.forward(batch[0],batch[1])       
            model.backward(gradReqd=True)
        train_loss,train_acc = modelTest(model,x_train,y_train)
        val_loss,val_acc = modelTest(model,x_valid,y_valid)
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        if early_stop:
            if val_loss>best_val_loss:
                patient+=1
                if patient == early_stop_epoch:
                    print(f"early_stop:{patient}")
                    end = time.time()
                    print(f"we trained {end -start} seconds for epoch #{epoch+1}")
                    break
            if val_loss<best_val_loss:
                best_val_acc = val_acc
                patient =0
                best_model = copy.deepcopy(model)
                best_val_loss = val_loss
        end = time.time()
        print(f"we trained {end -start} seconds for epoch #{epoch+1}")

    print("\n")
    print("\n")
    print(f"Best validation loss: {best_val_loss}")
    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Train Loss: {train_loss}")
    print(f"Train acc: {train_acc}")

    # visualizae the output
    plt.figure(figsize=(9,6))
    # plt.suptitle()
    plt.subplot(2,1,1)
    plt.xlabel("# of epochs")
    plt.ylabel("loss")
    plt.plot(train_losses, color='blue',label = "train loss")
    plt.plot(val_losses, color='red', label = "validation loss")
    plt.title("loss on validation and train")
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel("# of epochs")
    plt.ylabel("accuracy")
    plt.plot(train_accs, color='blue',label = "train accuracy")
    plt.plot(val_accs, color='red', label = "validation accuracy")
    plt.title("accuracy on validation and train")
    plt.legend()

    plt.tight_layout()
    plt.show(block=True)


    return best_model

#This is the test method
def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    return model.forward(X_test,y_test)


