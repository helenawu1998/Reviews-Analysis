import numpy as np
import os 
def load_data(filename, skiprows = 1):
	'''
	Function loads data stored in the file filename and returns it as a numpy ndarray.

	Inputs:
		filename: given as a string.

	Outputs:
		Data contained in the file, returns as a numpy ndarray.
	'''
	dir_path = os.path.dirname(os.path.realpath(filename)) + "\\" + filename
	data = np.loadtxt(filename, skiprows = skiprows, delimiter = ' ')
	return data
# Example usage of load_data if file includes label
# If file does not include label, test_data = load_data("test_data.txt")

''' 
filename = "training_data.txt"
data = load_data(filename)
x_input = data[:, 1:]
y_label = data[0:,0]
'''
#test_data = load_data("test_data.txt")