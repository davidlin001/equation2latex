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

class TranslationModel(object):
    def __init__(self):
        super(TranslationModel, self).__init__()
        self.encoder = encoder.build_encoder()
        # FILL IN INPUT INITIALIZATION FOR DECODER
        self.decoder = Decoder(HIDDEN_SIZE, OUTPUT_SIZE)

    def train(self, data):
        input_length = len(data)

        for i in range(input_length):
            encoded_input = self.encoder(img)
        
    
    def predict(self, img):
        pass



