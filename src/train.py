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

# Special vocabulary tokens and indexes
START = "<START>"
END = "<END>"
PAD = "<PAD>"
UNK = "<UNK>"
START_IDX = 0
END_IDX = 1
PAD_IDX = 2
UNK_IDX = 3

# Use GPU if available, otherwise use CPU
USE_CUDA = torch.cuda.is_available()

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
        optimizer : torch.optim.optimizer
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
    batch_size = kwargs.get("batch_size", 10)
    start_epoch = kwargs.get("start_epoch", 1)
    num_epochs = kwargs.get("num_epochs", 100)
    save_every = kwargs.get("save_every", 10)

    # Training loop
    for t in range(start_epoch, num_epochs + 1):
        
        print("Epoch {}...".format(t))

        model
            = train_on_batches(model, loss_fn, optimizer, train_dataset, 
                                batch_size, index_to_token, token_to_index)
        train_results 
            = eval_on_batches(model, train_dataset, metrics, batch_size, 
                                index_to_token, token_to_index)
        val_results 
            = eval_on_batches(model, val_dataset, metrics, batch_size, 
                                index_to_token, token_to_index) 

        train_results["loss"] = train_loss
        val_results["loss"] = val_loss

        # Update run statistics
        for metric, val in train_results.items():
            run_stats["train"][metric].append(float(val))
        for metric, val in val_results.items():
            run_stats["val"][metric].append(float(val))

        # Save checkpoint and plots
        if save_every != 0 and t % save_every == 0:
            save_checkpoint(model, optimizer, run_stats, t)
            plot_metric_scores(run_stats, t)


    return model

def save_checkpoint(model, optimizer, run_stats, epoch):
    """ Creates a checkpoint of the current |model|. The state dictionaries
    for the |model| and |optimizer|, the |run_stats| so far, and the current
    |epoch|, are saved to disk.

    Inputs:
        model : torch.nn.Module
            A torch model implementation that we would like train.
        optimizer : torch.optim.Optimizer
            The optimizer function we would like to use.
        run_stats : dict of defaultdicts
            Stores run metric values that we would like to track
            throughout training, such as the training loss,
            training and validation accuracies, etc. 
        epoch : int
            The current epoch number.

    Outputs:
        None, but a checkpoint is saved to disk.
    """
    state = (model.state_dict(), optimizer.state_dict(), run_stats, t)
    checkpoint_name = CKPT_PATH.substitute({"epoch" : epoch})
    torch.save(state, checkpoint_name)


def plot_metric_scores(run_stats, epoch):
    """ Plots the |run_stats| for each tracked metric up to the current |epoch|
    for both the training and validation sets. 

    Inputs:
        run_stats : dict of defaultdicts
            Stores run metric values that we would like to track
            throughout training, such as the training loss,
            training and validation accuracies, etc. 
        epoch : int
            The current epoch number.

    Outputs:
        None, but plots are saved to disk.
    """
    for dset in run_stats:
        for metric in run_stats[dset]:
            keywords = {"dset" : dset, "metric" : metric, "epoch" : epoch}
            plot_path = os.path.join(PLOT_PATH.substitute(keywords))
            plot_title = PLOT_TITLE.substitute(keywords)
            generate_time_series_plot(run_stats[dset][metric], plot_path, 
                                        ylabel=metric, title=plot_title)


def train_on_batches(model, loss_fn, optimizer, dataset, batch_size, index_to_token, token_to_index, optimize=True):
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
        loss_i = 0.0
        all_scores, all_indices = model(batch_images)
        max_length, batch_size = all_indices.shape
        for i in range(max_length):
            scores_i = all_scores[i, :, :]
            targets_i = [token_to_index.get(s, 3) for s in np.array(batch_formulas)[:,i]]
            targets_i = torch.LongTensor(targets_i).cuda()
            loss_i += torch.sum(loss_fn(scores_i, targets_i))
        loss += loss_i    

        # Backward pass
        optimizer.zero_grad()
        loss_i.backward()
        optimizer.step()

    # Average loss
    loss /= len(dataset)
    return model, loss


def eval_on_batches(model, dataset, metrics, batch_size, index_to_token, token_to_index):
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
        index_to_token : dict
            Maps integer IDs to unique vocabulary tokens.
        token_to_index : dict
            Maps vocabulary tokens to unique integer IDs.

    Outputs:
        results : dict
            Stores the computed metric scores on the given |dataset| for
            each metric in |metrics|.
    """

    # Process batches
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, collate_fn=collate_fn)
    for batch_images, batch_formulas in dataloader:  

        # Load batch
        if USE_CUDA:
            torch.cuda.empty_cache()
            batch_images = batch_images.cuda()

        # Get predicted tokens
        all_scores, all_indices = model(batch_images)
        all_tokens = all_indices.tolist()
        all_tokens = [[index_to_token[idx] for idx in ex] for ex in all_tokens]

        # Remove PAD tokens for metric computations
        batch_preds = [lst[:lst.index(END)+1] if END in lst else lst for lst in all_tokens]
        batch_formulas = [lst[:lst.index(END)+1] for lst in batch_formulas]

        # Compute metric values
        batch_results = collections.defaultdict(list)
        for mname, mfunc in metrics.items():
            batch_results[mname].append(mfunc(batch_preds, batch_formulas))
            
    # Average the results
    results = dict()
    for mname, mvals in batch_results.items():
        results[mname] = sum(mvals) / len(mvals)

    return results


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
    # Add special vocabulary tokens first
    index_to_token = {START_IDX: START, END_IDX: END, PAD_IDX: PAD, UNK_IDX: UNK}
    token_to_index = {START: START_IDX, END: END_IDX, PAD: PAD_IDX, UNK: UNK_IDX}

    # Add the rest of the vocab tokens
    with open(vocab_path,"r") as f:
        for token in f:
            token = token.strip()
            index_to_token[len(index_to_token)] = token
            token_to_index[token] = len(token_to_index)

    return index_to_token, token_to_index
        
    
if __name__ == "__main__":

    # Verify that everything is working as it should...
    
    # Run parameters
    MAX_LENGTH = 150
    EMBED_SIZE = 100
    SAVE_EVERY = 10

    # Filepaths to datasets
    images_path = "../data/sample/images_processed"
    formulas_path = "../data/sample/formulas.norm.lst"
    train_path = "../data/sample/train_filter.lst"
    validate_path = "../data/sample/validate_filter.lst"
    test_path = "../data/sample/test_filter.lst"
    vocab_path = "../data/sample/latex_vocab.txt"
    
    # Load the datasets
    train_dataset = Im2LatexDataset(images_path, formulas_path, train_path, MAX_LENGTH)
    val_dataset = Im2LatexDataset(images_path, formulas_path, validate_path, MAX_LENGTH)
    test_dataset = Im2LatexDataset(images_path, formulas_path, test_path, MAX_LENGTH)
    index_to_token, token_to_index = generate_vocab_mapping(vocab_path)

    # Try out the train function
    decoder_config = {
        "vocab_size": len(index_to_token), 
        "max_length": MAX_LENGTH, 
        "embed_size": EMBED_SIZE,
    }
    model = Im2Latex(decoder_config)
    model = model.cuda() if USE_CUDA else model
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    run_stats = {
        "train" : collections.defaultdict(list),
        "val" : collections.defaultdict(list),
    }
    kwargs = { 
        "save_every" : SAVE_EVERY
    }
    metrics = { 
        "edit_distance" : edit_distance_score 
    }
    model = train(model, loss_fn, optimizer, train_dataset, val_dataset, run_stats, metrics, index_to_token, token_to_index, **kwargs)

