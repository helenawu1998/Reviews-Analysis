print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2, SelectKBest
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# Chi square?
'''
ch2 = SelectKBest(chi2, k=100)
X_train = ch2.fit_transform(X_train, y_train)
X_test = ch2.transform(X_test)
'''

# ------------------------------------------------------------------------------
# PARAMETERS FOR CROSS-VALIDATION
k = 5
# ------------------------------------------------------------------------------

# Create LinearSVC.
clf = LogisticRegression(C=0.09)

# Fit model and get training error.
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
acc_val = np.mean(cross_val_score(clf, X_train, y_train, cv=k))
print('Validation accuracy: %g' % acc_val)

helper.process_output(clf.predict(X_test).astype(int), 'out/logreg.txt')
