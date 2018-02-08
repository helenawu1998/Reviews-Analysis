'''
Random forest model.
'''

print('Importing things...')
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import helper

# Load training and test datasets.
print('Loading training and test sets...')
X_train, y_train, X_test = helper.load_data('data/training_data.txt',
                                            'data/test_data.txt')

'''
# ------------------------------------------------------------------------------
# PARAMETERS FOR RANDOM FOREST CLASSIFIER
n_estimators = [1000]
max_depth = [None]
min_samples_leaf = [50]
max_features = [0.5]

# PARAMETERS FOR CROSS-VALIDATION
cv = 5
# ------------------------------------------------------------------------------

file = open('random_forest_optimizer.csv', 'a')
'''

'''
file.write('%s,%s,%s,%s,%s,%s\n' % ('acc_train',
                                    'acc_val',
                                    'n_estimators',
                                    'max_depth',
                                    'min_samples_leaf',
                                    'max_features'))
'''

'''
print('%s,%s,%s,%s,%s,%s' % ('acc_train',
                             'acc_val',
                             'n_estimators',
                             'max_depth',
                             'min_samples_leaf',
                             'max_features'))

for a in range(len(n_estimators)):
  for b in range(len(max_depth)):
    for c in range(len(min_samples_leaf)):
      for d in range(len(max_features)):

        # Create random forest classifier.
        rf = RandomForestClassifier(n_estimators=n_estimators[a],
                                    max_depth=max_depth[b],
                                    min_samples_leaf=min_samples_leaf[c],
                                    max_features=max_features[d])

        # Fit model and get training error.
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_train)
        acc_train = helper.accuracy(y_train, y_pred)

        # Get cross-validation error.
        acc_val = np.mean(cross_val_score(rf, X_train, y_train, cv=cv))

        file.write('%s,%s,%s,%s,%s,%s\n' % (acc_train,
                                            acc_val,
                                            n_estimators[a],
                                            max_depth[b],
                                            min_samples_leaf[c],
                                            max_features[d]))

        print('%s,%s,%s,%s,%s,%s' % (acc_train,
                                     acc_val,
                                     n_estimators[a],
                                     max_depth[b],
                                     min_samples_leaf[c],
                                     max_features[d]))

file.close()
'''

'''
# Create random forest classifier.
rf = RandomForestClassifier(n_estimators=n_estimators,
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                            max_features=max_features,
                            max_leaf_nodes=max_leaf_nodes,
                            min_impurity_decrease=min_impurity_decrease)
'''

# Create random forest classifier.
rf = RandomForestClassifier(n_estimators=100,
                            bootstrap=False,
                            criterion='entropy',
                            max_depth=None,
                            max_features=20,
                            min_samples_leaf=2,
                            min_samples_split=2)

# Fit model and get training error.
print('Training model...')
rf.fit(X_train, y_train)
y_pred = rf.predict(X_train)
print('Training accuracy: %g' % helper.accuracy(y_train, y_pred))

# Get cross-validation error.
print('Evaluating model...')
err_val = np.mean(cross_val_score(rf, X_train, y_train, cv=5))
print('Validation accuracy: %g' % err_val)

helper.process_output(rf.predict(X_test).astype(int), 'out/rand_forest_2.txt')

