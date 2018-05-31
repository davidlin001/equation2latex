# Title: utils.py
# Author: Cody Kala
# Date: 5/31/2018
# ===================
# Module for useful utility functions that do not belong in other modules.

import torch
import shutil
import os.path
import numpy as np
import matplotlib.pyplot as plt

def save_checkpoint(state, is_best, ckpt_file, best_ckpt_file):
	""" Saves the current |state| of the model to disk using the given
	filename.

	Inputs:
		state : tuple
			Contains information about the current state of the model.
			Includes the state dicts for the model and optimizer,
			the current epoch number, and a dictionary of various
			run statistics that are being tracked during training.
		is_best : boolean
			True if the current state of the model is the best model seen
			so far. Used to update the 
		ckpt_file : string
			The filename where the checkpoint is to be stored.
		best_ckpt_file : string
			The filename of the best checkpoint.

	Outputs:
		None, but saves the model checkpoint to disk.
	"""
	torch.save(state, ckpt_file)
	if is_best:
		shutil.copyfile(ckpt_file, best_ckpt_file)


def load_checkpoint(ckpt_file):
	""" Loads the model checkpoint stored at the given |ckpt_file|.

	Inputs:
		ckpt_file : string
			The filename where the checkpoint is stored.

	Outputs:
		state : tuple
			The current state of the model.
 
	"""

	if os.path.isfile(ckpt_file):
		print("==> loading checkpoint '{}'".format(ckpt_file))
		state = torch.load(ckpt_file)
		model_state_dict, optim_state_dict, run_stats, start_epoch = state
		print("==> loaded checkpoint '{}' (epoch {})"
				.format(ckpt_file, start_epoch))
		return state
	else:
		raise Exception("no checkpoint found at '{}'".format(ckpt_file))
		


def generate_time_series_plot(data, filename, xlabel="Epoch", ylabel="Data", 
								Title="Data vs. Time"):
	""" Generates a time series plot for the given |data|. The plot is
	stored at the directory pointed to by |filename|. This is essentially
	a convenience wrapper for plt.plot(). 

	Inputs:
		data : 1d np.array
			The data we would like to plot against iteration number.
		filename : string
			The location where we would like to save the plot.
		keyword arguments:
			xlabel : string
				The label for the x-axis of the plot.
				Default value is "Epoch".
			ylabel : string
				The label for the y-axis of the plot.
				Default value is "Data".
			title : string
				The title of the plot.
				Default value is "Data vs. Time".

	Outputs:
		None, but saves the plot to disk.
	"""
	time = np.arange(1, data.size + 1)
	plt.scatter(time, data, marker='x', color='red')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.savefig(filename)
	plt.clf()