print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# ------------------------------------------------------------------------------
# PARAMETERS FOR CROSS-VALIDATION
k = 5
# ------------------------------------------------------------------------------

# Create LinearSVC.
clf = ExtraTreesClassifier(n_estimators=300, max_features=20)

# Fit model and get training error.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
acc_val = np.mean(cross_val_score(clf, X_train, y_train, cv=k))
print('Validation accuracy: %g' % acc_val)

helper.process_output(clf.predict(X_test).astype(int), 'out/extra_trees.txt')
