# Title: train.py
# Author: Cody Kala
# Date: 5/31/2018
# ===================
# This module defines functions we use for training our learning models.

# Standard library imports
import torch
import collections
import os.path
import torch.optim as optim
import torch.nn as nn
from string import Template
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.functional import to_tensor
from PIL import Image
import sys
sys.path.insert(0, "./models/im2markup/")
sys.path.insert(0, "./models/im2latex/")

# Personal imports
from utils.dataset import Im2LatexDataset
from utils.utils import *
from models.im2markup.model import Im2Markup
from models.im2latex.model import Im2Latex
from metrics import *

# File directories
CKPT_DIR = "checkpoints"
CKPT_NAME = "checkpoint-epoch-${epoch}"
CKPT_PATH = Template(os.path.join(CKPT_DIR, CKPT_NAME))
PLOT_DIR = "plots"
PLOT_NAME = "${dset}-${metric}-${epoch}.png"
PLOT_PATH = Template(os.path.join(PLOT_DIR, PLOT_NAME))
PLOT_TITLE = Template("${dset} ${metric} vs epoch")

START = "<START>"
END = "<END>"
PAD = "<PAD>"
UNK = "<UNK>"

MAX_LENGTH = 50
EMBED_SIZE = 80
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
            metrics, index_to_token, token_to_index, **kwargs):
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
    batch_size = kwargs.get("batch_size", 20)
    start_epoch = kwargs.get("start_epoch", 1)
    num_epochs = kwargs.get("num_epochs", 20)
    save_every = kwargs.get("save_every", 10)
    split_ratio = kwargs.get("split_ratio", 0.8)

    # Training loop
    for t in range(start_epoch, num_epochs + 1):
        model = train_on_batches(model, loss_fn, optimizer, train_dataset, batch_size, index_to_token, token_to_index)
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


def train_on_batches(model, loss_fn, optimizer, dataset, batch_size, index_to_token, token_to_index):
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
        loss_i = 0.0
        # Load batch to GPU
        if USE_CUDA:
            torch.cuda.empty_cache()
            batch_images = batch_images.cuda()

        # Forward pass
        all_scores, all_indices = model(batch_images)
        #batch_preds = model(batch_images) 
        max_length, batch_size = all_indices.shape
        for i in range(max_length):
            scores_i = all_scores[i, :, :]
            targets_i = [token_to_index.get(s, 3) for s in np.array(batch_formulas)[:,i]]
            targets_i = torch.LongTensor(targets_i)
            loss_i += torch.sum(loss_fn(scores_i, targets_i))
            
        # Backward pass
        optimizer.zero_grad()
        loss_i.backward()
        optimizer.step()
        loss += loss_i
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
        formulas : list of list of string
            A minibatch of example formulas. The formulas tokenized
            and padded to max length.
    """
    images = [item[0] for item in data]
    images = standardize_dims(images)
    images = torch.stack(images, dim=0)
    formulas = [item[1] for item in data]
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

def generate_vocab_mapping(vocab_path):
    index_to_token = {0: START, 1: END, 2: PAD, 3:UNK}
    token_to_index = {START: 0, END: 1, PAD: 2, UNK:3}
    with open(vocab_path,"r") as f:
        idx = 4
        for token in f:
            token = token.strip()
            index_to_token[idx] = token
            token_to_index[token] = idx
            idx += 1
    return index_to_token, token_to_index
        
    
if __name__ == "__main__":

    # Verify that everything is working as it should...
    
    # Filepaths to datasets
    images_path = "../data/sample/images_processed"
    formulas_path = "../data/sample/formulas.norm.lst"
    train_path = "../data/sample/train_filter.lst"
    validate_path = "../data/sample/validate_filter.lst"
    test_path = "../data/sample/test_filter.lst"
    vocab_path = "../data/sample/latex_vocab.txt"
    # Load the datasets
    train_dataset = Im2LatexDataset(images_path, formulas_path, train_path)
    val_dataset = Im2LatexDataset(images_path, formulas_path, validate_path)
    test_dataset = Im2LatexDataset(images_path, formulas_path, test_path)


    index_to_token, token_to_index = generate_vocab_mapping(vocab_path)

    # Try out the train function
    decoder_config = {"vocab_size": len(index_to_token), "max_length": MAX_LENGTH, "embed_size": EMBED_SIZE}
    model = Im2Latex(decoder_config)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    run_stats = {
        "train" : collections.defaultdict(list),
        "val" : collections.defaultdict(list),
        "test" : collections.defaultdict(list),
    }
    kwargs = {}
    metrics = {"edit_distance": edit_distance_score}
    model = train(model, loss_fn, optimizer, train_dataset, val_dataset, run_stats, metrics, index_to_token, token_to_index, **kwargs)

