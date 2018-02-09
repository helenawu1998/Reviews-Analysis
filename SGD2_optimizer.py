'''
Finds optimal alpha to tune SVM model using SGD and log loss.

Model implements regularized linear models (SVM, logistic regression) with
stochastic gradient descent (SGD) learning

Optimal Alpha: alpha = 0.09
Highest validation accuracy:  0.8517
'''

print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')
# Pre-processing (Normalize Features)
means = []
sds = []
num_cols = X_train.shape[1]

# Normalize Training Set
for j in range(num_cols):
    mean = np.mean(X_train[:, j])
    means.append(mean)
    sd = np.std(X_train[:, j])
    sds.append(sd)
    X_train[:, j] = np.divide(np.subtract(X_train[:, j], mean), sd)

# Normalize Test Set (with the same mean and SD)
for j in range(num_cols):
    X_test[:, j] = np.divide(np.subtract(X_test[:, j], means[j]), sds[j])
# ------------------------------------------------------------------------------
# PARAMETERS FOR SVM with SGD
# loss = 'hinge'
loss = 'log'
penalty = 'l2'
# alpha = 0.09 # optimal alpha
tol = 0.00001
max_iter = 1000
shuffle = True
learning_rate = 'optimal'

# alphas = [.000001, .00001, .0001, .001, .01, .1, 1.0]
# alphas = [.01, .05, .09, .1, .11, .5, .9, 1.0]
alphas = [.06, .07, .08, .085, .09, .095, .1]

# PARAMETERS FOR CROSS-VALIDATION
k = 5
# ------------------------------------------------------------------------------

train_accs = []
val_accs = []
max_acc = 0
best_alpha = 0.0

for alpha in alphas:
    print('Modeling with alpha = ', alpha)
    # Create SVM with SGD model.
    clf = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, tol=tol,
    max_iter=max_iter, shuffle=shuffle, learning_rate=learning_rate)

    # Fit model and get training error.
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    train_accs.append(helper.accuracy(y_train, y_pred))

    # Get cross-validation error.
    err_val = np.mean(cross_val_score(clf, X_train, y_train, cv=k))
    val_accs.append(err_val)

    # Check if best validation accuracy
    if err_val >= max_acc:
        best_alpha = alpha
        max_acc = err_val

print('Optimal alpha: ', best_alpha)
print('Best validation accuracy: ', max_acc)
print(val_accs)
