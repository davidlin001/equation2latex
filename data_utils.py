# Title: utils.py
# Authors: Cody Kala, Max Minichetti, David Lin
# Date: 5/14/2018
# ================================
# This module implements useful utility functions we use to interact
# with the datasets.

import time
import os.path
import numpy as np
import cv2					# Used to read image data

# Directory to Kaggle dataset
KAGGLE_DATA_FILE = None

# Directory to the im2latex prebuilt datasets
IM2LATEX_PREBUILT_PATH = "im2latex-prebuilt-dataset"
IM2LATEX_IMAGES_PATH = os.path.join(IM2LATEX_PREBUILT_PATH, "formula_images")

# Name of im2latex prebuilt datasets
IM2LATEX_TRAIN_FILE = "im2latex_train.lst"
IM2LATEX_VALIDATE_FILE = "im2latex_validate.lst"
IM2LATEX_TEST_FILE = "im2latex_test.lst"
IM2LATEX_FORMULAS_FILE = "im2latex_formulas.lst"


def read_kaggle_data():
	""" Reads in the data from the Kaggle dataset. 

	Inputs:
		None

	Outputs:
		X : np.array of shape (num_examples, num_features)
			The i-th row corresponds to the i-th example in the dataset.
		y : np.array of shape (num_examples,)
			The i-th entry corresponds to the i-th example's label in the dataset.
	"""
	raise Exception("Not yet implemented.")

def _extract_im2latex_formulas(filepath):
	""" Private helper function that parses the formulas in the prebuilt im2latex datasets. 

	Inputs:
		filepath : string
			The path to the formulas file.

	Outputs:
		formulas: list
			The formulas for each image in the prebuilt dataset.
	"""
	with open(filepath, mode="r", encoding="ascii", errors="ignore", newline="\n") as f:
		formulas = f.readlines()
		formulas = [x.strip() for x in formulas]
	return formulas


def _extract_im2latex_dataset(filepath, formulas):
	""" Private helper function that parses the prebuilt im2latex datasets. 

	Inputs:
		filepath : string
			The path to the dataset file
		formulas : list
			The formulas for each image in the prebuilt dataset.
		
	Outputs:
		X : 
			Each row 

	"""
	X, y = [], []
	with open(filepath, mode="r") as f:
		for i, line in enumerate(f):
			formula_idx, image_name, __ = line.strip().split()
			if i % 10 == 0:
				print(formula_idx, image_name, __)
			formula_idx = int(formula_idx)	# convert to int
			formula = formulas[formula_idx]
			image_path = os.path.join(IM2LATEX_IMAGES_PATH, image_name + ".png")
			image =  cv2.imread(image_path)
			if i % 10 == 0:
				print(image.shape)
			X.append(image)
			y.append(formula)
	return X, y

def read_im2latex_data():
	""" Returns the datasets for im2latex dataset.

	Inputs:
		None

	Outputs:
		im2latex_datasets : dict
			Stores the training, validation, and test datasets for the im2latex
			datasets. As an example, to access the (X, y) pairs for the training
			set, we use the following:

				X_train = im2latex_datasets["train"]["X"]
				y_train = im2latex_datasets["train"]["y"]

			And similarly for the validation and test datasets.
	"""

	# Form the file paths
	formula_path = os.path.join(IM2LATEX_PREBUILT_PATH, IM2LATEX_FORMULAS_FILE)
	train_path = os.path.join(IM2LATEX_PREBUILT_PATH, IM2LATEX_TRAIN_FILE)
	val_path = os.path.join(IM2LATEX_PREBUILT_PATH, IM2LATEX_VALIDATE_FILE)
	test_path = os.path.join(IM2LATEX_PREBUILT_PATH, IM2LATEX_TEST_FILE)

	# Extract the data
	formulas = _extract_im2latex_formulas(formula_path)
	# X_train, y_train = _extract_im2latex_dataset(train_path, formulas)
	# X_val, y_val = _extract_im2latex_dataset(val_path, formulas)
	X_test, y_test = _extract_im2latex_dataset(test_path, formulas)

	# Store the datasets in a dictionary for easy lookup
	im2latex_datasets = {
		"train": {"X" : X_train, "y" : y_train},
		"val": {"X" : X_val, "y" : y_val},
		"test": {"X" : X_test, "y" : y_test},
	}
	return im2latex_datasets

if __name__ == "__main__":
	# Sanity checks
	tic = time.time()
	im2latex_dataset = read_im2latex_data()
	toc = time.time()
	print("Took {} seconds to read in data.".format(toc - tic))


