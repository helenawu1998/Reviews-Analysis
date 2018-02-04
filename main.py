import numpy as np
from sklearn.model_selection import KFold

import helper

# Load training and test datasets.
X_train, y_train, X_test = helper.load_data('data/training_data.txt', 'data/test_data.txt')

# Perform 5-fold cross-validation.
kf = KFold(n_splits = 5)
inds = [ind for ind in kf.split(X_train, y_train)]
train, val = inds[0]

'''
err_train = 0
err_val = 0

for i in range(len(inds)):
  train, val = inds[i]

  # Fit model on X_train[train], y_train[train]

  # Validate on X_train[val], y_train[val]
'''