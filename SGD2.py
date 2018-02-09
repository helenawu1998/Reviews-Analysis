'''
SVM model using SGD and log loss.

Implements regularized linear models (SVM, logistic regression) with
stochastic gradient descent (SGD) learning

Using alpha = 0.09
Highest validation accuracy:  0.8518494801405925
Best file: out/log_SGD2_3.txt
Val_accs: [0.8515994301280895, 0.851549392624962, 0.8516994676312167,
0.8518494801405925, 0.8515494676249666, 0.8516494176249635, 0.8513994676187167,
0.8516494176312136, 0.8514494176249636, 0.8515494176249636, 0.8515494926249682,
0.8513994176187136, 0.8515494176249636, 0.8514494426187152, 0.8513994301155894,
0.851599455128091]
Train_accs: [0.87185, 0.8717, 0.8716, 0.87165, 0.8718, 0.87185, 0.8718, 0.87165,
 0.87145, 0.8718, 0.87165, 0.8718, 0.8716, 0.87155, 0.8716, 0.87165]

Using alpha = 0.095
Highest validation accuracy:  0.851699330134333
Best file: out/log_SGD2_27.txt
Validation accuracies: [0.8512493676062105, 0.8511993676062104,
0.8513494176124636, 0.8514993551218346, 0.8510993675999605, 0.8514493676187105,
0.851299392612462, 0.8512493801093364, 0.8512494051093379, 0.8513994801218425,
0.8511994426062153, 0.851699330134333, 0.851299392612462, 0.8513494051155879,
0.8512493801093364, 0.8512993801093364]
Training accuracies: [0.8715, 0.8715, 0.8712, 0.8715, 0.87145, 0.8711, 0.87145,
 0.87135, 0.87145, 0.87145, 0.87135, 0.8714, 0.8714, 0.87145, 0.87125, 0.871]
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
alpha = 0.095
tol = 0.00001
max_iter = 1000
shuffle = True
learning_rate = 'optimal'


# PARAMETERS FOR CROSS-VALIDATION
k = 5

# ------------------------------------------------------------------------------
# Run SGD2 multiple (16) times with optimal alpha to get best output file
runs = [str(i) for i in range(16, 32)]
best_file = ''
best_val_acc = 0
val_accs = []
train_accs = []

for r in runs:
    print("Run Number: " + r)
    out_file = 'out/log_SGD2_' + r + '.txt'

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

    # Check for max validation accuracy.
    if err_val >= best_val_acc:
        best_val_acc = err_val
        best_file = out_file

    # Create output file in the format of sample_submission.
    helper.process_output(clf.predict(X_test).astype(int), out_file)

# Results from best run
print("Highest validation accuracy: ", best_val_acc)
print("Best file:", best_file)
print("Validation accuracies:", val_accs)
print("Training accuracies:", train_accs)
