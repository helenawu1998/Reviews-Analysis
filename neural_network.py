print("Importing packages")
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from keras import optimizers

from helper import load_data, process_output, error


train_file, test_file = "data/training_data.txt", "data/test_data.txt"
print("Loading data")
x_train, y_train, x_test = load_data(train_file, test_file)

# one-hot encode the labels
y_train = keras.utils.np_utils.to_categorical(y_train)

# normalize input data
x_train = np.divide(x_train, x_train.max())
x_test = np.divide(x_test, x_test.max())

# we must reshape the X data (add a channel dimension)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
x_test = x_test.reshape(tuple(list(x_test.shape) + [1]))

print("build model")
model = Sequential()
model.add(Flatten(input_shape=(1000,1)))  # Use np.reshape instead of this in hw
model.add(Dense(250))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))

model.add(Dense(250))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.3))


model.add(Dense(250))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='nadam', metrics=['accuracy'])
fit = model.fit(x_train, y_train, batch_size=128, nb_epoch=10,
verbose=1)
print("before process")
process_output(np.argmax(model.predict(x_test), axis = 1), "neural_network_predictions.txt")

'''
model.compile(loss='binary_crossentropy',optimizer='nadam', metrics=['accuracy'])
#fit = model.fit(x_train, y_train, batch_size=128, nb_epoch=10, verbose=1)
print("before keras classifier")
model = KerasClassifier(build_fn = model, epochs = 10, batch_size = 128, verbose = 1)
print("after")
kfold = KFold(n_splits = 10, shuffle = True)

print(np.mean(cross_val_score(model, x_train, y_train, cv = kfold)))
'''


print("hello")
