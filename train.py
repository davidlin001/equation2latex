# Title: train.py
# Author: Cody Kala
# Date: 5/31/2018
# ===================
# This module defines the training harness for our models.

import torch
import torchvision
import os.path
import numpy as np
import matplotlib.pyplot as plt

# File directories
CHECKPOINT_DIRECTORY = "checkpoints"
PLOTS_DIRECTORY = "plots"

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

def train(model, loss_fn, optimizer, features, labels, run_stats, **kwargs):
	""" This function trains the given |model| on the provided
	|features| and |labels|. The model parameters are optimized
	with respect to the |loss_fn| using the given |optimizer|.
	Aspects of the training harness, such as the start epoch,
	number of epochs, the batch size, etc. can be modified via
	the |kwargs| keyword dictionary.

	Inputs:
		model : torch.nn.Module
			A torch model implementation that we would like train.
		loss_fn : torch.nn.Module
			The loss function we would like optimizer our model on.
		optimizer : torch.optim.optimizer/
			The optimizer function we would like to use.
		features : torch.tensor of shape (num_examples, d1, d2, ...)
			The featurized inputs to our model. The i-th row
			corresponds to the i-th example in our training set.
		labels : torch.tensor of shape (num_examples,)
			The gold labels for our input features. The i-th entry
			corresponds to the i-th example in our training set.
		run_stats : dict
			Stores run statistics that we would like to track
			throughout training, such as the training loss,
			training and validation accuracies, etc.
		keyword arguments:
			batch_size : int
				The batch size to use for training
				Default value is 64.
			start_epoch : int
				The epoch to start on for training
				Default value is 1.
			num_epochs : int
				The number of epochs to train the model
				Default value is 20.
			save_every : int
				How often to save a checkpoint of the model (in epochs).
				Default value is 10.
			split_ratio : float between 0 and 1.
				Determines how the provided data is split into training
				and development sets. A larger split ratio indicates
				that more of the data should be used for training.
				Default value is 0.8 

	Outputs:
		model : torch.nn.Module
			The trained torch model.
	"""

	# Get keyword parameter values
	batch_size = kwargs.get("batch_size", 64)
	start_epoch = kwargs.get("start_epoch", 1)
	num_epochs = kwargs.get("num_epochs", 20)
	save_every = kwargs.get("save_every", 10)
	split_ratio = kwargs.get("split_ratio", 0.8)

	# Shuffle data
	num_examples = features.shape[0]
	perm = torch.randperm(num_examples)
	features, labels = features[perm], labels[perm]

	# Split data into training and development sets
	num_train = int(split_ratio * num_examples)
	train_features, train_labels = features[:num_train], labels[:num_train]
	dev_features, dev_labels = features[num_train:], labels[num_train:]

	# Training loop
	for t in range(start_epoch, num_epochs + 1):

		model, loss = train_on_batches(model, loss_fn, optimizer, 
									train_features, train_labels, batch_size)
		train_acc = eval_on_batches(model, train_features, train_labels, 
									batch_size)
		dev_acc = eval_on_batches(model, dev_features, dev_labels, 
									batch_size) 

		# Update run statistics
		run_stats["losses"].append(loss)
		run_stats["train_accs"].append(train_acc)
		run_stats["dev_accs"].append(dev_acc)

		##############################################
		#### TODO: Specify proper scoring metrics ####
		#### Raw accuracy is too harsh for our	  ####
		#### models... should use something a bit ####
		#### more holistic.  					  ####
		##############################################

		# Save checkpoint
		if t % save_every == 0:
			
			state = (
				model.get_state_dict(), 
				optimizer.get_state_dict(), 
				run_stats, 
				t,
			)
			checkpoint_name = os.path.join(CHECKPOINT_DIRECTORY,
									"checkpoint-epoch-{}.pth.tar".format(t))
			save_checkpoint(state, checkpoint_name)

			# Plot run statistics
			losses_name = os.path.join(PLOTS_DIRECTORY, 
									"losses-epoch-{}.png".format(t))
			train_accs_name = os.path.join(PLOTS_DIRECTORY, 
									"train-accs-epoch-{}.png".format(t))
			dev_accs_name = os.path.join(PLOTS_DIRECTORY, 
									"dev-accs-epoch-{}.png".format(t))
			generate_time_series_plot(run_stats["losses"], losses_name, 
									ylabel="Loss", 
									title="Loss vs. Epoch")
			generate_time_series_plot(run_stats["train_accs"], train_accs_name, 
									ylabel="Train Accuracy", 
									title="Train Accuracy vs. Epoch")
			generate_time_series_plot(run_stats["dev_accs"], dev_accs_name, 
									ylabel="Dev Accuracy",
									title="Dev Accuracy vs. Epoch")

	return model


def train_on_batches(model, loss_fn, optimizer, features, labels, batch_size):
	""" Trains the |model| for a single epoch on the provided |features| and
	|labels|. The model is optimized with respect to the |loss_fn| using the
	|optimizer|. Examples are processed in batches of size |batch_size|. 

	Inputs:
		model : torch.nn.Module
			A torch implementation of the model we would like to train.
		loss_fn : torch.nn.Module
			The loss function we optimize our loss function for.
		optimizer : torch.optim.optimizer
			The optimizer function we use to train our model.
		features : torch.tensor of shape (num_examples, d1, d2, ...)
			The featurized inputs to our model. The i-th row corresponds
			to the i-th example in the training set.
		labels : torch.tensor of shape (num_examples,)
			The gold labels for the featurized inputs. The i-th row corresponds
			to the i-th example in the training set.
		batch_size : int
			The number of examples to process in parallel.

	Outputs:
		model : torch.nn.Module
			The model, trained for a full epoch
		loss : float
			The average loss across all examples in the training set.
	"""
	
	# Initialize loss
	loss = 0.0

	# Process batches
	num_examples = features.shape[0]
	num_batches = max(num_examples // batch_size, 1)
	for i in range(num_batches):

		# Load batch
		start = batch_size * i
		end = min(batch_size * (i + 1), num_examples)
		batch_features = features[start: end]
		batch_labels = labels[start: end]
		if USE_CUDA:
			torch.cuda.empty_cache()
			batch_features = batch_features.cuda()
			batch_labels = batch_labels.cuda()

		# Forward pass
		batch_preds = model(batch_features)
		batch_loss = loss_fn(batch_preds, batch_labels)
		loss += batch_loss

		# Backward pass
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# Average loss
	loss /= num_examples
	return model, loss


def eval_on_batches(model, features, labels, batch_size):
	""" Evaluates the |model| on the provided |features| and |labels|.
	Examples are processed in batches of size |batch_size|.

	Inputs:
		model : torch.nn.Module
			A torch implementation of our model.
		features : torch.tensor of shape (num_examples, d1, d2, ...)
			The featurized inputs to our model. The i-th row corresponds
			to the i-th example in the dataset.
		labels : torch.tensor of shape (num_examples,)
			The gold labels for the featurized inputs. The i-th row corresponds
			to the i-th example in the dataset.
		batch_size : int
			The number of examples to process in parallel.

	Outputs:
		acc : float
			The accuracy of the model on the provided dataset. 
	"""
	
	# Initialize accuracy
	correct = 0
	total = 0

	# Process batches
	num_examples = features.shape[0]
	num_batches = max(num_examples // batch_size, 1)
	for i in range(num_batches):

		# Load batch
		start = batch_size * i
		end = min(batch_size * (i + 1), num_examples)
		batch_features = features[start: end]
		batch_labels = labels[start: end]
		if USE_CUDA:
			torch.cuda.empty_cache()
			batch_features = batch_features.cuda()
			batch_labels = batch_labels.cuda()

		# Score predictions
		batch_preds = model(batch_features)

		##############################################
		#### TODO: Specify proper scoring metrics ####
		#### Raw accuracy is too harsh for our	  ####
		#### models... should use something a bit ####
		#### more holistic.  					  ####
		##############################################

	acc = 0
	return acc