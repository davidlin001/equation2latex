# Title: model.py
# Author: Cody Kala
# Date: 6/3/2018
# ==================
# This Module implements the translation model introduced by Deng et al
# in this paper:
#
#   https://arxiv.org/pdf/1609.04938v2.pdf
#
# The authors have a GitHub with a TensorFlow implementation of their model
# as well. This can be found here:
#
#   https://github.com/harvardnlp/im2markup

import torch
import torch.nn as nn

from cnn import CNN
from encoder import Encoder
from decoder import Decoder

class Im2Markup(nn.Module):
    """ This module implements the translation model put forth by Deng et al.
    This model utilizes three distinct layers:

        1) A CNN layer to encode the raw images as feature maps of 
            shape (H, W, D), where D denotes the number of channels
            and H and W are the resulted feature map height and width.

        2) An Row Encoder layer that encoders the feature maps generated
            by the CNN layer into new feature maps by running an RNN
            across each row of the feature maps.

        3) A Decoder layer that generated the target markup tokens based
            only on the output of the Row Encoder layer.
    """

    def __init__(self, cnn_config, encoder_config, decoder_config):
        """ Initializes the layers of the Im2Markup model. 
        
        Inputs:
            cnn_config : dict
                A dictionary of keyword arguments used to configure the
                CNN layer.
            encoder_config : dict
                A dictionary of keyword arguments used to configure the
                Encoder.
            decoder_config : dict
                A dictionary of keyword arguments used to configure the
                Decoder.
        
        """
        self.cnn = CNN(**cnn_config)
        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)

    
    def forward(self, x):
        """ Computes the forward pass the Im2Markup model.

        Inputs:
            x : torch.Tensor of shape (num_batches, C, H, W)
                A mini-batch of images of LaTeX formulas.

        Outputs:
            out : torch.Tensor of shape (num_batches, max_length)
                A mini-batch of indexes into the vocabulary that can be decoded 
                (using a lookup dictionary) to generate the predicted LaTeX code.
        """
        out = self.cnn(x)
        out = self.encoder(x)
        out = self.decoder(x)
        return out
