# Title: utils.py
# Author: Cody Kala
# Date: 5/31/2018
# ===================
# Module for useful utility functions that do not belong in other modules.

def generate_time_series_plot(data, filename, xlabel="Epoch", ylabel="Data", 
								Title="Data vs. Time"):
	""" Generates a time series plot for the given |data|. The plot is
	stored at the directory pointed to by |filename|. This is essentially
	a convenience wrapper for plt.plot(). 

	Inputs:
		data : np.array or torch.tensor
			The data we would like to plot against iteration number.
		filename : string
			The location where we would like to save the plot.
		keyword arguments:
			xlabel : string
				The label for the x-axis of the plot.
				Default value is "Epoch".
			ylabel : string
				The label for the y-axis of the plot.
				Default value is "Data".
			title : string
				The title of the plot.
				Default value is "Data vs. Time".

	Outputs:
		None, but saves the plot to disk.
	"""
	raise NotImplementedError