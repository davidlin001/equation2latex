# Title: cnn.py
# Author: Cody Kala
# Date: 6/3/2018
# ==================
# Defines the CNN layer used in translation model proposed by Deng et al.

import torch
import torch.nn as nn

class CNN(nn.Module):
    """ Implements the Convolutional Network layer for the Im2Markup model
    proposed by Deng et al.
    """

    def __init__(self):
        """ Initializes the convolutional layers for the CNN.

        Inputs:
            ?

        Outputs:
            None
        """
        raise NotImplementedError


    def forward(self, x):
        """ Computes the forward pass for the CNN. """
        raise NotImplementedError
