from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.datasets import mnist
from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import to_categorical
import numpy as np

from cubic import Cubic

rho = 100
l2_reg = 0.01
c = 0.00001

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float64').reshape(x_train.shape[0], 28, 28, 1) / 255 - 0.5
x_test = x_test.astype('float64').reshape(x_test.shape[0], 28, 28, 1) / 255 - 0.5

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(4, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg), input_shape=(28, 28, 1, )))
model.add(MaxPooling2D())
model.add(Conv2D(16, kernel_size=2, padding='same', activation='relu', kernel_regularizer=l2(l2_reg),
    bias_regularizer=l2(l2_reg)))
model.add(AveragePooling2D())
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
model.add(Dense(32, activation='relu', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
model.add(Dense(10, activation='softmax', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg)))
model.compile(optimizer=Cubic(rho=rho, c=c), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=1024, epochs=10000,
    callbacks=[ModelCheckpoint(filepath='models/cubic&l=rho=%s&c=%s&l2=%s' % (rho, c, l2_reg)),
    TensorBoard(log_dir='logs/cubic&l=rho=%s&c=%s&l2=%s' % (rho, c, l2_reg))], validation_data=(x_test, y_test))
