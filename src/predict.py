# Title: predict.py
# Author: Cody Kala
# Date: 5/31/2018
# ====================
# This module defines functions we use for generating predictions on examples.

import torch
from torch.utils.data import DataLoader

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

def predict(model, dataset, batch_size):
    """ Use the trained |model| to predict the labels for the given |features|.
    Examples will be processed in batches of size |batch_size|.

    Inputs:
        model : torch.tensor.Module
            The trained model, implemented in PyTorch.
        dataset : torch.util.data.Dataset
            A Dataset instance containing the example images for which we
            would like to predict LaTeX formulas. 
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
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch_images, batch_formulas in range(num_batches):

        # Load batch onto GPU
        if USE_CUDA:
            torch.cuda.empty_cache()
            batch_images = batch_images.cuda()

        # Predict batch labels
        batch_preds = model(batch_images))
        preds.append(batch_preds)

    # Concatenate predictions
    preds = torch.cat(preds)
    return preds
