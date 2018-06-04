# Title: main.py
# Author: Cody Kala
# Date: 5/31/2018
# =================
# This module is where the functions in other modules are called in order
# to train and develop our learning models.

# Standard library imports
import torch
import torch.nn as nn

# Personal imports
from utils.dataset import *
from utils.utils import *
from models.im2latex import *
from models.im2markup import *
from train import *
from predict import *

# Filepaths to im2latex data
images_path = "data/full/images_processed"
formulas_path = "data/full/formulas.norm.lst"
train_path = "data/full/train_filter.lst"
val_path = "data/full/validate_filter.lst"
test_path = "data/full/test_filter.lst"

def main():
    # Load datasets
    train_dataset = Im2LatexDataset(images_path, formulas_path, train_path)
    val_dataset = Im2LatexDataset(images_path, formulas_path, val_path)
    test_dataset = Im2LatexDataset(images_path, formulas_path, test_path)

    # Initialize 
    model = TranslationModel()
    optim = torch.optim.Adam()
    loss_fn = nn.CrossEntropyLoss()
    run_stats = {} 

    # Run experiments
    mode = input("mode? [train/test]")
    checkpoint_path = input("path to checkpoint? Enter "" to use fresh model")
    if mode == "train":
        if checkpoint_path != "" 
            state = load_checkpoint(checkpoint_path)
            model.load_state_dict(state["model_state_dict"])
            optim.load_state_dict(state["optim_state_dict"])
            run_stats.update(state["run_stats"])
            train_kwargs["start_epoch"] = state["start_epoch"]
        model = train(model, optim, loss_fn, run_stats, train_dataset, 
                        val_dataset, train_kwargs)
    if mode == "test":
        if checkpoint_path != ""
            state = load_checkpoint(checkpoint_path)
            model.load_state_dict(state["model_state_dict"])
            optim.load_state_dict(state["optim_state_dict"])
            run_stats.update(state["run_stats"])
            train_kwargs["start_epoch"] = state["start_epoch"]
    

if __name__ == "__main__":
    main()
