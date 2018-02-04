'''
Random forest model.
'''

print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# ------------------------------------------------------------------------------
# PARAMETERS FOR RANDOM FOREST CLASSIFIER
n_trees = 100
criterion = 'gini'
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0.0
max_features = 'auto'
max_leaf_nodes = None
min_impurity_decrease = 0.0

# PARAMETERS FOR CROSS-VALIDATION
k = 5
# ------------------------------------------------------------------------------

# Create random forest classifier.
rf = RandomForestClassifier(n_estimators=n_trees,
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                            max_features=max_features,
                            max_leaf_nodes=max_leaf_nodes,
                            min_impurity_decrease=min_impurity_decrease)

# Fit model and get training error.
print('Training model...')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
print('Evaluating model...')
err_val = np.mean(cross_val_score(rf, X_train, y_train, cv=k))
print('Validation accuracy: %g' % err_val)

helper.process_output(rf.predict(X_test).astype(int), 'out/rand_forest_1.txt')
