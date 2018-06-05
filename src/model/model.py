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
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import encoder
from decoder import Decoder

class TranslationModel(nn.Module):
    def __init__(self):
        super(TranslationModel, self).__init__()
        self.encoder = encoder.build_encoder()
        # FILL IN INPUT INITIALIZATION FOR DECODER
        self.decoder = Decoder(20, 10)

    def forward(self, input):
        pass
