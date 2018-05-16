# Title: baseline_model.py
# Authors: Cody Kala, Max Minichetti, David Lin
# Date: 5/14/2018
# ===========================
# Defines a baseline model for our CS231n final project.

import numpy as np
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import classification_report, confusion_matrix  

K = 1
X = []
y = [] 

# Path to data directory
data_dir = 'baselineData/KaggleSymbolData/extracted_images/'
labels = os.listdir(data_dir)

for label in labels:
    # Ignore hidden directories
    if label[0] == '.':
        continue
    count = 0 
    for filename in os.listdir(data_dir + label):
        count += 1
        # Sets limit on number of training examples from each label
        if count > 500:
            break
        # Convert image to numpy array
        pic_array = np.asarray(Image.open(data_dir + label + '/' + filename))
        X.append(pic_array.flatten())
        y.append(label)
    print("Done with ", label)

# Training with 80-20 Split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20) 
classifier = KNeighborsClassifier(n_neighbors=K)  
classifier.fit(X_train, y_train)  
y_pred = classifier.predict(X_test)  
print("Prediction Accuracy ", np.average(y_test == y_pred))
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))
'''
class BaselineModel(TorchModelBase):

	def __init__(self):
		""" Stores the parameters that are relevant for the model """
		raise Exception("Not yet implemented.")

	def fit(self, X, y):
		""" Trains the model on the training examples (X, y) """
		raise Exception("Not yet implemented.")

	def predict(X):
		""" Predicts the labels for the examples in X """
		raise Exception("Not yet implemented.")


if __name__ == "__main__":
	# Sanity checks for the model
	raise Exception("Not yet implemented.")

'''
