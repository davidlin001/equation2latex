# Title: main.py
# Author: Cody Kala
# Date: 5/31/2018
# =================
# This module is where the functions in other modules are called in order
# to train and develop our learning models.

import torch
import torchvision
from model import *
from train import *
from predict import *
from utils import *


# Filepaths to Im2Latex data
images_path = "im2latex/images"
train_formulas = "im2latex/train_formulas.lst"
train_lookup = "im2latex/train_lookup.lst"
val_formulas = "im2latex/val_formulas.lst"
val_lookup = "im2latex/val_lookup.lst"
test_formulas = "im2latex/test_formulas.lst"
test_lookup = "im2latex/test_lookup.lst"

#########################################################
#### TODO: Implement basic UI functionality for main ####
#########################################################

def main():

    # Setup datasets
    train_dataset = Im2LatexDataset(images_path, train_formulas, train_lookup)
    val_dataset = Im2LatexDataset(images_path, val_formulas, val_lookup)
    test_dataset = Im2LatexDataset(images_path, test_formulas, test_lookup)

    # Initialize
    model = TranslationModel()
    optimizer = torch.optim.Adam()
    loss_fn = torch.nn.CrossEntropyLoss()
    

    # Train/test harness
    mode = input("mode? [train/test]")
    if mode == "train":
        use_checkpoint = input("load from checkpoint? [yes/no]")
        if use_checkpoint == "yes":
            checkpoint_name = input("path to checkpoint? ")
        elif use_checkpoint == "no":
            model

if __name__ == "__main__":
    main()
