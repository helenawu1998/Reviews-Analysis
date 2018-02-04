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

	return np.loadtxt(filename, skiprows = skiprows, delimiter = ' ')
filename = "training_data.txt"
dir_path = os.path.dirname(os.path.realpath(filename)) + "\\" + filename
x = load_data(dir_path)