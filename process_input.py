import numpy as np
import os 
def load_data(train_data, test_data, skiprows = 1):
	'''
	Function loads training and test data stored in input files in the same folder as load_data
	and returns x_train, y_train, and x_test in numpy ndarrays.

	Inputs:
		train_data: training_data filename
		test_data: test_data filename

	Outputs:
		x_train: x values for training set as numpy ndarray
		y_train: labels for x values in training set as numpy ndarray
		x_test: x values for testing set as numpy ndarray
	'''
	
	train_data = np.loadtxt(train_data, skiprows = skiprows, delimiter = ' ')

	x_train = train_data[:, 1:]
	y_train = train_data[0:,0]

	x_test = np.loadtxt(test_data, skiprows = skiprows, delimiter = ' ')

	return x_train, y_train, x_test

# Example usage of load_data
'''
x_train, y_train, x_test = load_data("training_data.txt", "test_data.txt")
'''