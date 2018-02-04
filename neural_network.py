import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Flatten, BatchNormalization
from keras import regularizers

from helper import load_data, process_output, error

#def neural_network(train_file, test_file):

train_file, test_file = "data/training_data.txt", "data/test_data.txt"

x_train, y_train, x_test = load_data(train_file, test_file)

# one-hot encode the labels
y_train = keras.utils.np_utils.to_categorical(y_train)

# normalize input data
x_train = np.divide(x_train, x_train.max())
x_test = np.divide(x_test, x_test.max())

# we must reshape the X data (add a channel dimension)
x_train = x_train.reshape(tuple(list(x_train.shape) + [1]))
x_test = x_test.reshape(tuple(list(x_test.shape) + [1]))

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)

model = Sequential()
model.add(Flatten(input_shape=(1000,1)))  # Use np.reshape instead of this in hw
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.20))

model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
fit = model.fit(x_train, y_train, batch_size=128, nb_epoch=10,
verbose=1)

process_output(np.argmax(model.predict(x_test), axis = 1), "out/neural_network_predictions.txt")

#neural_network("training_data.txt", "test_data.txt")
#print("hello")
