# SynaNet - Synaptic Neural Network

A Synaptic Neural Network (SynaNN) consists of synapses and neurons. Inspired by the research of biological synapses, we abstracted a synapse as a nonlinear function of two excitatory and inhibitory probabilities

SynaMPL for Keras illustrates the application of Synaptic Neural Network to Multiple Layer Perceptrons. Synaptic Multiple Layer Perceptrons is the fully-connected Multiple Layer Perceptron (Minsky et al. (2017)) with two hidden layers connected by a synapse tensor along with an input layer and an output layer. Input and output layers act as downsize and classifiers while hidden layer of the synapse tensor plays the role of distribution transformation. The activation functions are required to connected


Figure 1: SynaMLP: (green, blue, red) dots are (input, hidden, output) layers.

SynaMLP for Keras.

'''Trains a simple SynaNet MLP on the MNIST dataset.

    Copyright Rights (c) 2018, Neatware.

    Open Source License: Apache 2.0
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD

from keras import backend as K
from keras.engine.topology import Layer

import tensorflow as tf

batch_size = 100
num_classes = 10
epochs = 30
hidden_size = 250
onedim = 784
traindim = 60000
testdim = 10000

# synapse unit
class Synapse(Layer):
  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(Synapse, self).__init__(**kwargs)

  def build(self, input_shape):
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)
    super(Synapse, self).build(input_shape)

  # def syna_block(self, xx, ww, MM, batch):
  def syna_block(self, xx):
    MM = self.output_dim
    batch = batch_size
    ww = self.kernel
    shapex = tf.reshape(tf.matrix_diag(xx), [-1,MM])
    ww2 = tf.transpose(ww)
    # ww2 = tf.transpose(tf.matrix_set_diag(ww, tf.zeros(MM, tf.float32)))
    # ww2 = tf.matrix_set_diag(ww, tf.zeros(MM, tf.float32))
    betax = tf.log(1.0-tf.matmul(shapex, ww2))
    allone = tf.ones([batch, batch*MM], tf.float32)
    return tf.exp(tf.log(xx) + tf.tensordot(allone, betax, 1))

  def call(self, x):
    return self.syna_block(x)

  def compute_output_shape(self, input_shape):
     return (input_shape[0], self.output_dim)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(traindim, onedim)
x_test = x_test.reshape(testdim, onedim)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# sequential
model = Sequential()

# 1. input layer
model.add(Dense(hidden_size, input_shape=(onedim,), trainable=True))
model.add(keras.layers.normalization.BatchNormalization())
model.add(Activation('softmax'))

# 2. hidden layer
model.add(Synapse(hidden_size))
model.add(keras.layers.normalization.BatchNormalization())

# 3. output layer
model.add(Dense(num_classes, activation='softmax'))

# summary
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])

# history
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# evaluate(x=None, y=None, batch_size=None, verbose=
score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

