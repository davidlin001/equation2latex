# Title: model.py
# Author: David Lin, Cody Kala
# Date: 5/31/2018
# =============================
# Implements the translation model of Guillaume Genthial and Romain Sauvestre
# for generating LaTeX code from images of mathematical expressions.
#
# The original paper can be found here:
#
#   cs231n.stanford.edu/reports/2017/pdfs/815.pdf
#
# Their GitHub, which contains a TensorFlow implementation of the model,
# can be found here:
#
#  https://github.com/guillaumegenthial/im2latex

import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Im2Latex(nn.Module):
    """ A PyTorch implementation of the translation model proposed by
    Guillaume Genthial and Romain Sauvestre for the Im2Latex-100k
    dataset.
    """
    
    ####################################################################
    #### TODO: Edit signatures to include other necessary arguments ####
    ####################################################################

    def __init__(self, encoder_config, decoder_config):
        """ Initializes the stages of the Im2Latex model. 
        
        Inputs:
            encoder_config : dict
                A dictionary of keyword arguments used to
                configure the Encoder.
            decoder_config : dict
                A dictionary of keyword arguments used to
                configure the Decoder.

        Outputs:
            None, but initializes the layers.
        """
        super().__init__()
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)


    def forward(self, x):
        """ Computes the forward pass for the Im2Latex model. 
        
        Inputs:
            x : torch.Tensor of shape (batch_size, C, H, W)
                A minibatch of images all of size (C, H, W), where

                    C is the number of channels,
                    H is the number of pixels in the height dimension
                    W is the number of pixels in the width dimension

        Outputs:
            out : torch.Tensor of shape (batch_size, max_length)
                A mini-batch of idxs into the vocabulary that that can
                be decoded to determine the predicted LaTeX.
        """
        out = self.encoder(x)
        out = self.decoder(out)
        return out

