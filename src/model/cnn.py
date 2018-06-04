# Title: cnn.py
# Author: Cody Kala
# Date: 6/3/2018
# ===================
# This Module implements the initial CNN layer that transforms the raw
# images into a feature grid to be used by the encoder. This architecture
# is based on the work of Deng et al. Their Arxiv paper can be found here:
#
#   https://arxiv.org/pdf/1609.04938.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """ The CNN module describes the deep convolutional neural network
    layer that is used to process the raw images and convert them into
    feature grids, one for each image.

    Each feature grid has shape D x H x W, where D denotes the number
    of channels and H and W are the resulted feature map height and
    width. 
    """

    def __init__(self):
        """ Initializes the layers to be used in the CNN. 

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
