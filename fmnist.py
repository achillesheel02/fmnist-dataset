from __future__ import print_function, division
from builtins import range
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import matplotlib.pylab as plt
import numpy as np


# the one-hot encoder
def y2indicator(y):
    n = len(y)
    k = len(set(y))
    i = np.zeros((n, k))
    i[np.arange(n), y] = 1
    return i


data = pd.read_csv('fashion-mnist_train.csv')
data = data.as_matrix()
np.random.shuffle(data)

# getting the labels and data
X = data[:, 1:].reshape(-1, 28, 28, 1) / 255.0
Y = data[:, 0].astype(np.int32)

K = len(set(Y))

Y = y2indicator(Y)

model = Sequential()

model.add(Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Conv2D(filters=128, kernel_size=(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(units=300))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(units=K))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

r = model.fit(X, Y, validation_split=0.33, epochs=15, batch_size=32)
print("Returned ", r)

print(r.history.keys())

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

plt.plot(r.history['acc'], label='acc')
plt.plot(r.history['val_acc'], label='val_acc')
plt.legend()
plt.show()
