# Title: train.py
# Author: Cody Kala
# Date: 5/31/2018
# ===================
# This module defines functions we use for training our learning models.

# Standard library imports
import torch
import os.path
from string import Template
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from PIL import Image

# Personal imports
from utils.dataset import Im2LatexDataset
from utils.utils import *
from models.im2markup import Im2Markup
from models.im2latex import Im2Latex
from metrics import *

# File directories
CKPT_DIR = "checkpoints"
CKPT_NAME = "checkpoint-epoch-${epoch}"
CKPT_PATH = Template(os.path.join(CKPT_DIR, CKPT_PATH))
PLOT_DIR = "plots"
PLOT_NAME = "${dset}-${metric}-${epoch}.png"
PLOT_PATH = Template(os.path.join(PLOT_DIR, PLOT_PATH))
PLOT_TITLE = Template("${dset} ${metric} vs epoch")

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

###########################################################################
#### TODO: Find a way to process images of similar sizes. Currently,   ####
#### images will be placed into mini-batches randomly, but often times ####
#### there will be at least one image that has the largest dimension,  ####
#### o we end up training having standardize the images to that size   ####
#### to process the images as a batch. If we can somehow group the     ####
#### images inte buckets as suggested by the papers, then this would   ####
#### improve training and evaluatio efficiency considerably.           ####
###########################################################################


def train(model, loss_fn, optimizer, train_dataset, val_dataset, run_stats, 
            metrics, **kwargs):
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
        run_stats : dict of defaultdicts
            Stores run metric values that we would like to track
            throughout training, such as the training loss,
            training and validation accuracies, etc. 
            
            run_stats should be a dictionary with three keys: "train",
            "val", and "test". The value of each key is a defaultdict of lists,
            used to store the values of metrics computed during the course of
            training and evaluation. 
            
            The keys to the defaultdicts will be the names of the metrics, and 
            the values will be lists storing the computed values.
        
        metrics : dict of callbacks
            Stores the metrics we would like to evaluate our model on. The
            keys are the names of the metrics, and the values are the callables
            used to evaluate a mini-batch of examples at onces.

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

        model = train_on_batches(model, loss_fn, optimizer, train_dataset, batch_size)
        train_results = eval_on_batches(model, train_dataset, metrics, batch_size)
        val_results = eval_on_batches(model, val_dataset, metrics, batch_size) 

        # Update run statistics
        for metric, val in train_results.items():
            run_stats["train"][metric].append(val)
        for metric, val in val_results.items():
            run_stats["val"][metric].append(val)

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
            # TODO: Code below should be factored into a utility function
            for metric in metrics:
                train_keywords = { 
                    "dset" : "train", 
                    "metric" : metric, 
                    "epoch" : t 
                }
                train_plot_path = os.path.join(PLOT_PATH.substitute(keywords))
                train_plot_title = PLOT_TITLE.substitute(keywords)
                generate_time_series_plot(run_stats["train"][metric], 
                                          train_plot_path,
                                          ylabel=metric,
                                          title=train_plot_title)
    
                val_key_words = { 
                    "dset" : "val", 
                    "metric" : metric, 
                    "epoch" : t,
                }
                val_plot_path = os.path.join(PLOT_PATH.substitute(keywords))
                val_plot_title = PLOT_TITLE.substitue(keywords)
                generate_time_series_plot(run_stats["val"][metric],
                                          val_plot_path,
                                          ylabel=metric,
                                          title=val_plot_title)


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
        batch_preds = model(batch_features)  
        batch_loss = loss_fn(batch_preds, batch_formulas)
        loss += batch_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Average loss
    loss /= len(dataset)
    return model, loss


def eval_on_batches(model, dataset, metrics, batch_size):
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
    or evaluation on minibatches. This function is passed to the DataLoader 
    in the collate_fn keyword argument.

    Inputs:
        data : list
            A list of examples we would like to train or evaluate our 
            model on. Examples should be of the form (image, formula), 
            where |image| is a torch tensor representing the raw pixel
            data of the image, and |formula| is a string representing
            the LaTeX formula. 
            
            The images in |data| may not all be the same size, so to account 
            for this we must standardize across the spatial dimensions. 

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
    """ Adds padding to the images in |images| so that all images in the batch have
    the same dimensions. This allows the batch of images to be stacked into a single torch
    tensor and passed to the model at once during batch training and batch evaluation.

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
