'''
Decision tree model with AdaBoost.
'''

print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# ------------------------------------------------------------------------------
# PARAMETERS FOR DECISION TREE CLASSIFIER
max_leaf_nodes = 40

# PARAMETERS FOR ADABOOST
n_clfs = 100

# PARAMETERS FOR CROSS-VALIDATION
cv = 5
# ------------------------------------------------------------------------------

# Create decision tree classifier.
dt = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)

# Create AdaBoost classifier.
ada = AdaBoostClassifier(dt, n_estimators=n_clfs)

# Train model and get training error.
print('Training model...')
ada.fit(X_train, y_train)
y_pred = ada.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
print('Evaluating model...')
acc_val = np.mean(cross_val_score(ada, X_train, y_train, cv=cv))
print('Validation accuracy: %g' % acc_val)

# helper.process_output(rf.predict(X_test), 'decision_tree.txt')
