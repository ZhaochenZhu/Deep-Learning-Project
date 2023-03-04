import argparse
import network
import data
import numpy as np
from matplotlib import pyplot as plt
import time



def select_model(model_dictionary):
	"""
	model_dictionary:
		 a dictionary of dictionaries where each inner dictionary
	store the data for a model including train/val loss, accuracy
	"""
	best_acc = float("-inf")
	for model in model_dictionary:
		if model_dictionary[model]['best_acc'] > best_acc:
			best_model = model_dictionary[model]['network']
			best_acc = model_dictionary[model]['best_acc']
		else:
			continue
	return best_model, best_acc


def filter(first,second,data,label,normalized_function):
	X=[]
	Y=[]
	for i in range(len(label)):
		# if i%10000 ==0:
		# 	print(i) 
		if label[i] == first or label[i] == second:
			X.append(data[i])
			Y.append(label[i])
	for i in range(len(Y)):
		if Y[i]==second:
			Y[i] =1
		else:
			Y[i] =0
	for i in range(len(X)):
		X[i] = normalized_function(X[i])[0]
	return np.array(X),np.array(Y)

def main(hyperparameters):
	main_start = time.time()
	#import and shuffle data
	train_data,train_label = data.load_data()
	train_data,train_label = data.shuffle((train_data,train_label))
	test_data,test_label = data.load_data(train=False)
	test_data,test_label = data.shuffle((test_data,test_label))

	# considering binary for 0 and 5 only
	# will need one more argument to tell in the future
	if hyperparameters.z_score =="z_score":
		normalized_function = data.z_score_normalize
	else:
		normalized_function = data.min_max_normalize
	
	# filter the data, normalize them
	if hyperparameters.binary ==1:
		train_X,train_Y = filter(0,5,train_data,train_label,normalized_function)
		test_X,test_Y = filter(0,5,test_data,test_label,normalized_function)
		train_dataset = (train_X,train_Y)
	if hyperparameters.binary ==-1:
		train_X,train_Y = filter(3,5,train_data,train_label,normalized_function)
		test_X,test_Y = filter(3,5,test_data,test_label,normalized_function)
		train_dataset = (train_X,train_Y)
	if hyperparameters.binary == 0:
		train_X = train_data
		test_X = np.array(test_data)
		for i in range(len(train_X)):
			train_X[i] = normalized_function(train_X[i])[0]
		for i in range(len(test_X)):
			test_X[i] = normalized_function(test_X[i])[0]
		train_dataset = (np.array(train_X),np.array(train_label))
		test_Y = np.array(test_label)
	
	
	#record best accuracies
	best_accs = []
	
	# Cross Validation
	fold_count = 1
	networks = {}
	folds = data.generate_k_fold_set(train_dataset)
	epochs = hyperparameters.epochs
	batch_size = hyperparameters.batch_size

	lr = hyperparameters.learning_rate

	for train_set, val_set in folds:
		start = time.time()
		print("fold #%i..." % fold_count)
		#create a database for each network
		network_dict = {}

		#for each fold, store data for each epochs
		epoch_losses = [] #store train loss
		epoch_acc = [] #store train acc
		epoch_val_losses = [] #store val loss
		epoch_val_acc = [] #store val acc

		best_epoch = 1
		best_acc = float('-inf')
		best_loss = float('inf')

		#initialize network for each fold
		if hyperparameters.binary ==1 or hyperparameters.binary ==-1:
			cur_Network = network.Network({'lr':lr},network.sigmoid,network.binary_cross_entropy,out_dim=1)
		else: 
			cur_Network = network.Network({'lr':lr},network.softmax,network.multiclass_cross_entropy,out_dim=10)
		network_dict['network'] = cur_Network

		for epoch in range(epochs):
			train_set = data.shuffle(train_set)
			batches = data.generate_minibatches(train_set,batch_size=batch_size)


			#train model
			j=0
			for batch in batches:
				cur_Network.train(batch)
			#evaluate on validation
			loss_on_val, acc_on_val = cur_Network.test(val_set)
			loss_on_train,acc_on_train = cur_Network.test(train_set)

			epoch_losses.append(loss_on_train)
			epoch_acc.append(acc_on_train)
			epoch_val_losses.append(loss_on_val)
			epoch_val_acc.append(acc_on_val)

			#early stop
			if loss_on_val<=best_loss:
				best_loss = loss_on_val
				best_acc = acc_on_val
				best_epoch = epoch
				best_weight = cur_Network.weights
			else:
				cur_Network.weights = best_weight
				#break
			# best_loss = loss_on_val
			# best_acc = acc_on_val
			# best_weight = cur_Network.weights


		print(f"\t we run {epoch+1} epochs\n" )

		best_accs.append(best_acc)
		network_dict['best_acc'] = best_acc
		network_dict['train_loss'] = epoch_losses
		network_dict['train_acc'] = epoch_acc
		network_dict['val_loss'] = epoch_val_losses
		network_dict['val_acc'] = epoch_val_acc
		networks[fold_count] = network_dict
		end = time.time()
		print(f"we trained {end -start} seconds for fold #{fold_count}")
		fold_count += 1

	# print("best validation accs = ", best_accs)
	# print("avg validation acc = ", np.mean(best_accs))
	# print()

	
	# two ways to get the best model: 
	# 1. use the one I stored in networks(already trained) 
	# 2. create a new network and assign the weights stored


	Best_Network, best_acc = select_model(networks)
	print("best validation accs = ", best_accs)
	print("avg validation accs = ", np.mean(best_accs))
	Best_test_loss,Best_test_accuracy = Best_Network.test((test_X,test_Y))
	print("Best model test loss = ", Best_test_loss)
	print("Best model test accuracy = ", Best_test_accuracy)

	
	# print the average loss of the best model
	total_train_loss = np.zeros((1,100))[0]
	total_val_loss = np.zeros((1,100))[0]
	total_train_acc = np.zeros((1,100))[0]
	total_val_acc = np.zeros((1,100))[0]
	for model in networks:
		list_train_acc = networks[model]['train_acc']
		list_val_acc = networks[model]['val_acc']
		list_train_loss = networks[model]['train_loss']
		list_val_loss = networks[model]['val_loss']
		while len(list_val_acc)<100:
			list_val_acc.append(list_val_acc[-1])
		while len(list_train_acc)<100:
			list_train_acc.append(list_train_acc[-1])
		while len(list_train_loss)<100:
			list_train_loss.append(list_train_loss[-1])
		while len(list_val_loss)<100:
			list_val_loss.append(list_val_loss[-1])
		total_train_loss += np.array(list_train_loss)
		total_val_loss += np.array(list_val_loss)
		total_train_acc += np.array(list_train_acc)
		total_val_acc += np.array(list_val_acc)
	total_train_loss/= 10
	total_train_acc /= 10
	total_val_loss /= 10
	total_val_acc /= 10

	
	print(f"the best model is on the {best_epoch} epoch")
	main_end = time.time()
	print(f"main run {(main_end - main_start)/60} minutes")


	

	plt.figure(figsize=(9,6))
	if hyperparameters.binary == 1:
		plt.suptitle(f"Logistic training for airplane and dog, batchsize = {hyperparameters.batch_size}, learning rate = {hyperparameters.learning_rate}, normalization = {hyperparameters.z_score}")
	if hyperparameters.binary ==-1:
		plt.suptitle(f"Logistic training for cat and dog, batchsize = {hyperparameters.batch_size}, learning rate = {hyperparameters.learning_rate}, normalization = {hyperparameters.z_score}")
	if hyperparameters.binary == 0:
		plt.suptitle(f"Softmax multiclass classification, batchsize = {hyperparameters.batch_size}, learning rate = {hyperparameters.learning_rate}, normalization = {hyperparameters.z_score}")

	plt.subplot(2, 1, 1)
	plt.plot(total_train_loss, color='blue',label = "train loss")
	plt.xlabel("# of epochs")
	plt.ylabel("loss")
	plt.plot(total_val_loss, color='red', label = "validation loss")
	plt.title("loss on validation and train")
	plt.legend()

	plt.subplot(2,1,2)
	plt.plot(total_train_acc, color = 'blue',label = "train accuracy")
	plt.plot(total_val_acc,color = 'red', label = "validation accuracy")
	plt.xlabel('# of epoch')
	plt.ylabel("accuracy")
	plt.title('accuracy on validation and train')
	plt.legend()
	
	plt.tight_layout()
	plt.show(block=True)



parser = argparse.ArgumentParser(description = 'CSE151B PA1')

parser.add_argument("--binary", type = int, default = 1)
parser.add_argument('--batch_size', type = int, default = 1,
        help = 'input batch size for training (default: 1)')
parser.add_argument('--epochs', type = int, default = 100,
        help = 'number of epochs to train (default: 100)')
parser.add_argument('--learning_rate', type = float, default = 0.001,
        help = 'learning rate (default: 0.001)')
parser.add_argument('--z_score',type = str, default = 'z_score')
parser.add_argument('--k_fold', type = int, default = 10,help = 'number of folds for cross-validation')
# parser.add_argument('--z_score',dest = 'normalization', action='store_const',default = data.min_max_normalize, const = data.z_score_normalize,help = 'use z-score normalization on the dataset, default is min-max normalization')


hyperparameters = parser.parse_args()
main(hyperparameters)