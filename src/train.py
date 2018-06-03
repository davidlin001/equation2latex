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
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from PIL import Image
from dataset import Im2LatexDataset
from metrics import *

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

        print("Training...")
        model, loss = train_on_batches(model, loss_fn, optimizer, 
                                    train_dataset, batch_size)
        print("Evaluating on training set...")
        train_acc = eval_on_batches(model, train_dataset, batch_size)
        print("Evaluating on validation set...")
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


def train_on_batches(model, loss_fn, optimizer, dataset, batch_size):
    """ Trains the |model| for a single epoch on the provided |dataset|. 
    The model is optimized with respect to the |loss_fn| using the
    |optimizer|. Examples are processed in batches of size |batch_size|. 

    Inputs:
        model : torch.nn.Module
            A torch implementation of the model we would like to train.
        loss_fn : torch.nn.Module
            The loss function we optimize our loss function for.
        optimizer : torch.optim.optimizer
            The optimizer function we use to train our model.
        dataset : torch.utils.data.Dataset
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
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, collate_fn=collate_fn) 
    for batch_images, batch_formulas in dataloader:

        # Load batch to GPU
        if USE_CUDA:
            torch.cuda.empty_cache()
            batch_images = batch_images.cuda()

        # Forward pass
        # batch_preds = model(batch_images)  
        # batch_loss = loss_fn(batch_preds, batch_formulas)
        # loss += batch_loss

        # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    # Average loss
    loss /= len(dataset)
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
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)
    for batch_images, batch_formulas in dataloader:
    
        # Load batch
        if USE_CUDA:
            torch.cuda.empty_cache()
            batch_images = batch_images.cuda()

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


def collate_fn(data):
    """ Collates the items in the |data| in preparation for training
    or evaluation. This is passed to the DataLoader function in the
    collate_fn keyword argument.

    Inputs:
        data : list
            A list of examples we would like to train or evaluate our 
            model on. Examples should be of the form (image, formula), 
            where |image| is a torch tensor representing the raw pixel
            data of the image, and |formula| is the LaTeX formula
            associated to the image.

    Outputs:
        images : torch.tensor of shape (batch_size, C, H, W) 
            A minibatch of example images.
        formulas : list of strings
            A minibatch of example formulas.
    """
    images = [item[0] for item in data]
    formulas = [item[1] for item in data]
    images = standardize_dims(images)
    images = torch.stack(images, dim=0)
    return images, formulas


def standardize_dims(images):
    """ Adds padding to the images in |images| so that the images have the same
    dimensions. This allows the images to stack for the training and evaluation
    steps on mini-batches.

    Inputs:
        images : list of torch tensors
            Contains torch tensors representing the images in the batch.

    Outputs:
        new_images : list of torch tensors
            Contains torch tensors representing the images in the batch. The dimensions
            of the images are the same.
    """
    # Find largest dimension
    __, max_width, max_height = images[0].size()
    for image in images:
        __, width, height = image.size()
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

    # Pad images to these new dimensions
    new_images = []
    for image in images:
        pil_image = to_pil_image(image)
        new_pil_image = Image.new("RGB", (max_width, max_height), color=255)
        new_pil_image.paste(pil_image, pil_image.getbbox())
        new_image = to_tensor(new_pil_image)
        new_images.append(new_image)

    return new_images  


if __name__ == "__main__":

    # Verify that everything is working as it should...
    
    # Filepaths to datasets
    images_path = "../data/full/images_processed"
    formulas_path = "../data/full/formulas.norm.lst"
    train_path = "../data/full/train_filter.lst"
    validate_path = "../data/full/validate_filter.lst"
    test_path = "../data/full/test_filter.lst"

    # Load the datasets
    train_dataset = Im2LatexDataset(images_path, formulas_path, train_path)
    val_dataset = Im2LatexDataset(images_path, formulas_path, validate_path)
    test_dataset = Im2LatexDataset(images_path, formulas_path, test_path)

    # Try out the train function
    model = None
    optimizer = None
    loss_fn = None
    run_stats = { "losses" : [], "train_accs" : [], "val_accs" : [] }
    kwargs = { "save_every" : 0, "num_epochs" : 1}
    model = train(model, loss_fn, optimizer, train_dataset, val_dataset, run_stats, **kwargs)
