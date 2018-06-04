# Title: encoder.py
# Author: David Lin, Cody Kala
# Date: 6/3/2018
# ===================
# This Module implements the encoder stage of the translation
# model used by Guillaume Genthial and Romain Sauvestre in
# their paper, which can be found here:
#
#   cs231n.stanford.edu/reports/2017/pdfs/815.pdf  

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """ The Encoder uses a deep convolutional neural network to
    encode the images. The Encoder encodes the original image
    of size H x W into a feature map of size H' x W' x C, where
    C is the number of filters of the last convolutional layer.

    The Encoder defines one vector v_{h', w'} for h' in {1, ..., H'}
    and w' in {1, ..., W'}. Intuitively, each v_{h', w'} captures
    the information from one region of the image.
    """

    def __init__(self):
        """ Initializes the layers to be used in the Encoder.

        Inputs:
            None

        Outputs:
            None, but the layers are initialized.
        """
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64,
                               kernel_size = 3, padding = 1, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128,
                               kernel_size = 3, padding = 1, stride = 1)
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 256,
                               kernel_size = 3, padding = 1, stride = 1)
        self.conv4 = nn.Conv2d(in_channels = 256, out_channels = 256,
                               kernel_size = 3, padding = 1, stride = 1)
        self.conv5 = nn.Conv2d(in_channels = 256, out_channels = 512,
                               kernel_size = 3, padding = 1, stride = 1)
        self.conv6 = nn.Conv2d(in_channels = 512, out_channels = 512,
                               kernel_size = 3, padding = 1, stride = 1)
        
        # Initialize max pooling layer
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        """ Implements the forward pass for the Encoder. 

        Inputs:
            x : torch.tensor of shape (batch_size, C, H, W)
                A minibatch of images.

        Outputs:
            out : torch.tensor of shape (batch_size, 512, ?, ?)
                The encodings for the input images.
        """
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = F.relu(out)

        out = self.conv4(out)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv5(out)
        out = F.relu(out)
        out = self.pool(out)

        out = self.conv6(out)
        out = F.relu(out)
        return out
