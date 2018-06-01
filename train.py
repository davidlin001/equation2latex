# Title: train.py
# Author: Cody Kala
# Date: 5/31/2018
# ===================
# This module defines functions we use for training our learning models.

import torch
import torchvision
import os.path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Im2LatexDataset

# File directories
CHECKPOINT_DIRECTORY = "checkpoints"
PLOTS_DIRECTORY = "plots"

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

def train(model, loss_fn, optimizer, train_dataset, val_dataset, run_stats, **kwargs):
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
        train_dataset : torch.utils.data.Dataset
            A Dataset instance containing the training images and their LaTeX
            formulas.
        val_dataset : torch.utils.data.Dataset
            A Dataset instance containing the validation images and their
            LaTeX formulas.
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
                To turn off saving, set this value to 0.
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

    # Training loop
    for t in range(start_epoch, num_epochs + 1):

        model, loss = train_on_batches(model, loss_fn, optimizer, 
                                    train_dataset, batch_size)
        train_acc = eval_on_batches(model, train_dataset, batch_size)
        val_acc = eval_on_batches(model, val_dataset, batch_size) 

        # Update run statistics
        run_stats["losses"].append(loss)
        run_stats["train_accs"].append(train_acc)
        run_stats["val_accs"].append(val_acc)

        ##############################################
        #### TODO: Specify proper scoring metrics ####
        #### Raw accuracy is too harsh for our    ####
        #### models... should use something a bit ####
        #### more holistic.                       ####
        ##############################################

        # Save checkpoint
        if save_every != 0 and t % save_every == 0:
            
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
            val_accs_name = os.path.join(PLOTS_DIRECTORY, 
                                    "val-accs-epoch-{}.png".format(t))
            generate_time_series_plot(run_stats["losses"], losses_name, 
                                    ylabel="Loss", 
                                    title="Loss vs. Epoch")
            generate_time_series_plot(run_stats["train_accs"], train_accs_name, 
                                    ylabel="Train Accuracy", 
                                    title="Train Accuracy vs. Epoch")
            generate_time_series_plot(run_stats["val_accs"], val_accs_name, 
                                    ylabel="Validation Accuracy",
                                    title="Validation Accuracy vs. Epoch")

    return model


def train_on_batches(model, loss_fn, optimizer, train_dataset, batch_size):
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
        train_dataset : torch.utils.data.Dataset
            A Dataset instance containing the training images and their
            LaTeX formulas.
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
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    for batch_images, batch_formulas in dataloader:
        
        # Load batch to GPU
        if USE_CUDA:
            torch.cuda.empty_cache()
            batch_images = batch_images.cuda()
            batch_formulas = batch_formulas.cuda()  # May not work since |batch_formulas| is tuple

        # Forward pass
        # batch_preds = model(batch_features)  
        # batch_loss = loss_fn(batch_preds, batch_formulas)
        # loss += batch_loss

        # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    # Average loss
    loss /= len(train_dataset)
    return model, loss


def eval_on_batches(model, dataset, batch_size):
    """ Evaluates the |model| on the provided |features| and |labels|.
    Examples are processed in batches of size |batch_size|.

    Inputs:
        model : torch.nn.Module
            A torch implementation of our model.
        dataset : torch.utils.data.Dataset
            A Dataset instance containing the images and their LaTeX
            formulas.
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
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for batch_images, batch_formulas in dataloader:
    
        # Load batch
        if USE_CUDA:
            torch.cuda.empty_cache()
            batch_images = batch_images.cuda()
            batch_formulas = batch_formulas.cuda()

        # Score predictions
        # batch_preds = model(batch_images)

        ##############################################
        #### TODO: Specify proper scoring metrics ####
        #### Raw accuracy is too harsh for our    ####
        #### models... should use something a bit ####
        #### more holistic.                       ####
        ##############################################

    acc = 0
    return acc


if __name__ == "__main__":

    # Verify that everything is working as it should...
    
    # Filepaths to datasets
    images_path = "im2latex/images"
    train_formula_path = "im2latex/train_formulas.lst"
    train_lookup_path = "im2latex/train_lookup.lst"
    val_formula_path = "im2latex/val_formulas.lst"
    val_lookup_path = "im2latex/val_lookup.lst"
    test_formula_path = "im2latex/test_formulas.lst"
    test_lookup_path = "im2latex/test_lookup.lst"

    # Load the datasets
    train_dataset = Im2LatexDataset(images_path, train_formula_path, train_lookup_path)
    val_dataset = Im2LatexDataset(images_path, val_formula_path, val_lookup_path)
    test_dataset = Im2LatexDataset(images_path, test_formula_path, test_lookup_path)

    # Try out the train function
    model = None
    optimizer = None
    loss_fn = None
    run_stats = { "losses" : [], "train_accs" : [], "val_accs" : [] }
    kwargs = { "save_every" : 0 }
    model = train(model, loss_fn, optimizer, train_dataset, val_dataset, run_stats, **kwargs)
