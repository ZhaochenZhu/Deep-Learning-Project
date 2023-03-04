# Image Classification with backpropagation and experiments on the hyper parameters
In this project, we've train our model using the data from cifar-10, we've tested different hyperparameters(learning_rate, momentum, regularization, activation used, number of hidden units and number of hidden layers) in different experiments. When running the program, the output will generate the graph to show you the loss and accuracy on the training and validation set, as well as the performance on test set. The best parameters have been tested and saved in the config file for each experiment. To run the program, just run main.py with an argument to specify which experiment we are doing. 

## Package used
### numpy
We use numpy to do do matrix operation and make our code more simplified and readable
### os/pickle
We use os and pickle to get data from cifar-10 dataset
### time
We use time to keep track of the runtime of each epoch and the whole program
### matplotlib
We use matplotlib.pyplot to visualize our training performance

## How to run the program:
Command below will read the parameters from the <config_name>.yaml files in config folder, initialize and train model with given parameters.

### test_gradients config_2b
```bash
python main.py --experiment "test_gradients"
```
Check if a specific weight in Neural Network is updated correctly.
Change the layer and weight index in gradient.py to specify weight to check.

### test_momentum config_2c
```bash
python main.py --experiment "test_momentum"
```
Read and train model with hyper-parameters in config/config_2c.yaml.
"learning_rate" config_2c.yaml can be change to any reasonable positive numerical value for the model.

### test_regularization config_2d
```bash
python main.py --experiment "test_regularization"
```
Read and train model with hyper-parameters in config/config_2d.yaml.
Change "L2_penalty" in config_2d.yaml to test effect of different regularization parameters.

### test_activation config_2e
```bash
python main.py --experiment "test_activation"
```
Read and train model with hyper-parameters in config/config_2e.yaml.
"acitivation" in config_2e.yaml can be changed to acitivation function among "sigmoid", "tanh", "ReLU" for all hidden layers in the model.

### test_hidden_units config_2f_i
```bash
python main.py --experiment "test_hidden_units"
```
Read and train model with hyper-parameters in config/config_2f_i.yaml.
Change "layer_specs" config_2f_i.yaml. Change the middel value of the layer list to test different number of node in the hidden layer. (ie \[3073, node number, 10\])

### test_hidden_layers config_2f_ii
```bash
python main.py --experiment "test_hidden_layers"
```
Read and train model with hyper-parameters in config/config_2f_ii.yaml.
Change "layer_specs" config_2f_ii.yaml. Change the values in layer list except the first and last element to train model of different number of layers. example:\[3073, layer1_node_number, layer2_nodes_number,..., 10\]

<!-- 1. main.py is the driver code that has to be run in order to run different set of experiments
2. run get_cifar10data.sh to download the data. The dataset will be downloaded in 'data' directory 
3. config files need to be in the 'config' directory
4. You are free to create new functions, change existing function
signatures, add/remove member variables/functions in the provided classes but you need to maintain the overall given structure 
of the code. Specifically, you have to mandatorily use the classes provided in neuralnet.py to create your model (although, 
like already mentioned, you can  add or remove variables/functions in these classes or change the function signatures)
5. We have marked sections to be implemented by you as TODO for your convenience -->

<!-- ToDo:
Clarification Question:
1. In 2c, the only hyper we can change is lr and batchsize? (i.e. keep # of hidden layers, dimension of hidden layer unchanged)
    Yes, only lr should change. Also 42% is fine
2. In 2d, where we need to test on the regularization, do we need to use momentum as well? If so, how? (high level idea, math formula)
    We can use momentum if it provide better performance. (generally we should use momentum)
    Add one term to weight update
3. Do we have a desired accuracy for 2d,e,f
    >44% would be fine -->
