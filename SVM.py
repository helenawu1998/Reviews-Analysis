'''
SVM model using support vector clustering (SVC).
Generally bad with large datasets and large number of feature.

Can possibly use feature selection/dimensionality reduction with SVM.
'''

print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# ------------------------------------------------------------------------------
# PARAMETERS FOR SVC
c = 1.0
kernel = 'linear'
deg = 3
coeff = 0.0
gamma = 1.0

# PARAMETERS FOR CROSS-VALIDATION
k = 5
# ------------------------------------------------------------------------------

# Create SVC.
clf = SVC(C=c, kernel=kernel, degree=deg, coef0=coeff, gamma=gamma)

# Fit model and get training error.
print('Training model...')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
print('Evaluating model...')
err_val = np.mean(cross_val_score(clf, X_train, y_train, cv=k))
print('Validation accuracy: %g' % err_val)

helper.process_output(clf.predict(X_test).astype(int), 'SVM_out.txt')
