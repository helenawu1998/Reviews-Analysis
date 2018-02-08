print('Importing things...')
import numpy as np
from time import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GridSearchCV
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# Build a classifier.
clf = BernoulliNB()

# Utility function to report best scores.
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score: {0:.3f} (std: {1:.3f})'.format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print('Parameters: {0}'.format(results['params'][candidate]))
            print('')

# Use a full grid over all parameters.
param_grid = {'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              'fit_prior': [True, False]}

# Run grid search.
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train, y_train)

print('GridSearchCV took %.2f minutes for %d candidate parameter settings.'
      % ((time() - start) / 60, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
