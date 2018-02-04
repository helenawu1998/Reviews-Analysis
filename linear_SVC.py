'''
SVM model using linear support vector clustering (LinearSVC).

Similar to SVC with kernel='linear' but implemented in terms of liblinear
rather than libsvm so it is more flexible and scalable with large samples.
'''

print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# ------------------------------------------------------------------------------
# PARAMETERS FOR LinearSVC
penalty = 'l2'
loss = 'hinge'
c = 1.0
tol = 0.0001

# PARAMETERS FOR CROSS-VALIDATION
k = 5
# ------------------------------------------------------------------------------

# Create LinearSVC.
clf = LinearSVC(penalty=penalty, loss=loss, C=c, tol=tol)

# Fit model and get training error.
print('Training model...')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
print('Evaluating model...')
err_val = np.mean(cross_val_score(clf, X_train, y_train, cv=k))
print('Validation accuracy: %g' % err_val)

helper.process_output(clf.predict(X_test).astype(int), 'LinearSVC_out.txt')
