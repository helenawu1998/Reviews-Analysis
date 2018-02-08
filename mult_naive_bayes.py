'''
Multinomial Naive Bayes
'''

print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# ------------------------------------------------------------------------------
# Parameters for multinomial naive bayes
alpha = 0.9

# Parameters for cross-validation
k = 5
# ------------------------------------------------------------------------------

# Create SVM with SGD model.
clf = MultinomialNB(alpha=alpha)

# Fit model and get training error.
print('Training model...')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
print('Evaluating model...')
acc_val = np.mean(cross_val_score(clf, X_train, y_train, cv=k))
print('Validation accuracy: %g' % acc_val)

# helper.process_output(clf.predict(X_test).astype(int), 'out/naive_bayes.txt')
