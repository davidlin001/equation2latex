# Title: torch_model_base.py
# Authors: Cody Kala, Max Minichetti, David Lin
# Date: 5/14/2018
# =============================

class TorchModelBase(object):
	""" Simple interface to which all of our PyTorch models will confirm. """

	def __init__(self):
		pass

	def fit(self, X, y):
		""" Fits the model to the training examples (X, y)

		Inputs:
			X : np.array of shape (num_examples, num_features)
				The i-th row corresponds to the features of the i-th example.

			y: np.array of shape (num_examples,)
				The i-th entry corresponds to the label of the i-th example.

		Outputs:
			None, but trains the model.
		"""
		pass

	def predict(self, X):
		""" Predicts the outputs for the examples supplied in |X|.

		Inputs:
			X : np.array of shape (num_examples, num_features)
				The i-th row corresponds to the features of the i-th example.

		Outputs:
			pred : np.array of shape (num_examples,)
				The i-th entry corresponds the predicted label for the i-th example.
		"""
		pass

