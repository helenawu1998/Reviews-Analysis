print('Importing things...')
import numpy as np
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# Build a classifier.
clf = RandomForestClassifier(n_estimators=100)

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
param_grid = {'max_depth': [None],
              'max_features': [20, 21, 22, 23],
              'min_samples_leaf': [2, 3, 4, 5],
              'criterion': ['entropy']}

# Run grid search.
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X_train, y_train)

print('GridSearchCV took %.2f minutes for %d candidate parameter settings.'
      % ((time() - start) / 60, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
