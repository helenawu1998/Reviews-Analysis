'''
Majority vote on 4 different models.
'''

print('Importing things...')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

# Create classifiers.
rf1 = RandomForestClassifier(n_estimators=100, max_features=20)
rf2 = RandomForestClassifier(n_estimators=100, max_features='auto')
# svc = SVC(C=1.0, kernel='linear', degree=3, coef0=0.0, gamma=1.0)
sgd1 = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, tol=0.00001,
                     max_iter=1000, shuffle=True, learning_rate='optimal')
sgd2 = SGDClassifier(loss='log', penalty='l2', alpha=0.09, max_iter=1000,
                     shuffle=True, learning_rate='optimal')
nb1 = GaussianNB()
nb2 = BernoulliNB()

# Create voting classifier.
vc = VotingClassifier(estimators=[('rf1', rf1), ('rf2', rf2), ('sgd1', sgd1),
                                  ('sgd2', sgd2), ('nb1', nb1), ('nb2', nb2)])

# Fit model and get training error.
print('Training model...')
vc.fit(X_train, y_train)
y_pred = vc.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
print('Evaluating model...')
acc_val = np.mean(cross_val_score(vc, X_train, y_train, cv=5))
print('Validation accuracy: %g' % acc_val)

helper.process_output(vc.predict(X_test).astype(int), 'out/majority3.txt')
