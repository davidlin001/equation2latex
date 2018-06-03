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

def build_encoder():
    return nn.Sequential(
        # 4 CNN/ Max Pool, CNN/ Max pool/ CNN/ Output
        # Check in_channels on first one
        nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 3, padding = 1, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1, stride = 1),
        nn.ReLU(),
        nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1, stride = 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
        nn.ReLU(),
        # Can replace this and the previous max pooling layer with a cnn of stride 2
        nn.MaxPool2d(kernel_size = 2, stride = 2),
        nn.Conv2d(in_channels = 512, out_channels = 512, kernel_size = 3, padding = 1, stride = 1),
        nn.ReLU()
    )
