# Title: predict.py
# Author: Cody Kala
# Date: 5/31/2018
# ====================
# This module defines functions we use for generating predictions on examples.

import torch

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

def predict(model, features, batch_size):
    """ Use the trained |model| to predict the labels for the given |features|.
    Examples will be processed in batches of size |batch_size|.

    Inputs:
        model : torch.tensor.Module
            The trained model, implemented in PyTorch.
        features : torch.Tensor of shape (num_examples, d1, d2, ...)
            The featurized examples whose labels we would like to predict.
            The i-th row contains the featurized input for the i-th examples
        batch_size : int
            The number of examples the model should process in parallel.

    Outputs:
        preds : torch.Tensor shape (num_examples,)
            The predicted labels for the input features. The i-th row
            contains the predicted label for the i-th example.
    """
    
    # Initialize
    preds = []

    # Process in batches
    num_examples = features.shape[0]
    num_batches = max(num_examples // batch_size, 1)
    for i in range(num_batches):

        # Load batch
        start = batch_size * i
        end = min(batch_size * (i + 1), num_examples)
        batch_features = features[start: end]
        if USE_CUDA:
            batch_features = batch_features.cuda()

        # Predict batch labels
        batch_preds = model(batch_features)
        preds.append(batch_preds)

    # Concatenate predictions
    preds = torch.cat(preds)
    return preds
