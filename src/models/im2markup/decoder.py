# Title: decoder.py
# Author: David Lin, Cody Kala
# Date: 6/3/2018
# ===============================
# This module implements the decoder stage of the translation model pipeline.

import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    """ This module implements the decoder stage of our translation model.
    It uses some variant of a recurrent neural network to predict a
    distribution over the vocabulary of LaTeX symbols.
    """

    def __init__(self, hidden_size, output_size, cell_type="lstm"):
    	""" Initializes the decoder module.

    	Inputs:
    		hidden_size : int
    			The dimension of the hidden state vector
			output_size : int
				The dimension of the output state vector
			cell_type : string
				Optional, specifies the type of RNN cell to use. Must be 
				one of "rnn", "gru", or "lstm". The default value is "lstm".

		Outputs:
			None
		"""
		super().__init__()

		# Initialize embedding vector
		self.embedding = nn.Embedding(output_size, hidden_size)

		# Initialize the RNN cell
		if cell_type == "rnn":
			self.cell = nn.RNN(hidden_size, hidden_size)
		elif cell_type == "gru":
			self.cell = nn.GRU(hidden_size, hidden_size)
		elif cell_type == "lstm":
			self.cell = nn.LSTM(hidden_size, hidden_size)
		else:
			raise Exception("Unexpected cell_type. Must be one of 'rnn', 'gru', or 'lstm.'")


	def forward(self, x, hidden):
		""" Computes the forward pass for the decoder stage. """
		out = self.embedding(x)
		out = F.relu(out)
		out, hidden = self.cell(out, hidden)
		out = F.log_softmax(out, dim=1)
		return out, hidden

##############################
#### TODO: Fix this up... ####
##############################


    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        #Option 1: Naive RNN captioning
        #self.gru = nn.GRU(hidden_size, hidden_size)
        #Option 2: LSTM captioning
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        #Option 1: RNN
        #output, hidden = self.gru(output, hidden)
        #Option 2: LSTM
        output, hidden = self.LSTM(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



