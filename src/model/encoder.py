# Title: encoder.py
# Author: David Lin, Cody Kala
# Date: 6/3/2018
# ===============================
# Implements the encoder stage of our translation model pipeline.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
	""" The Encoder class encapsulates a series of convolutional layers
	followed by ReLU and Max Pooling layers. The output of the Encoder
	is a collection of "annotation" vectors that encode information about
	various patches of the images. 
	"""

	def __init__(self):
		# Initialize convolutional layers
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

