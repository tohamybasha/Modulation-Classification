#import keras
#import tensorflow as tf

#import torch
#import torch.nn as nn

import pickle
import numpy as np

''' 
    the data set has 11 modulations
    for each modulation we have 20 snr
    for each modulation/snr we have 1000 instance
    ************************************************************************
                                for example 
    8 PSK modulation has 20 snr values from -20 to 18 with step value = 2
    for each value of the snr for 8 PSK modulation we have 1000 example 
    and the same for each modulation 
    so for every modulation we have 20(snrs)*1000=20000 example
    ************************************************************************
    the dataset =11 (modulation)* 20000(example for each modulation)=220000 example

'''
#a = open("RML2016.10b/RML2016.10b.dat", 'rb')
a = open("RML2016.10a/RML2016.10a_dict.pkl", 'rb')
u = pickle._Unpickler(a)
u.encoding = 'latin1'
Xd = u.load()

# Xd = pickle.load(a)
snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
X = []
lbl = []
print(len(mods))
print(len(snrs))
print(mods)
print(snrs)
print(len(mods) * len(snrs))
clasees = np.zeros(shape=220000)
dataset_snrs = np.zeros(shape=220000)
j = 0
classes_index = 0

mod_to_int = {}
for mod in mods:
    j += 1
    print('mod = ', mod)
    mod_to_int[j] = mod

    for snr in snrs:
        '''
        classes is the labels dataset each modulation will have a number from 1 to 11 
        mod to int is to map between number and clasees 
        dataset_snrs is all snr as a one vector to be more easier to deal with 

        '''

        clasees[classes_index:classes_index + 1000] = j
        dataset_snrs[classes_index:classes_index + 1000] = snr
        # print('snr ',snr)
        classes_index += 1000
        # print(Xd[mod,snr].shape)
        X.append(Xd[(mod, snr)])
        for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
X = np.vstack(X)
print(X.shape)
# print(len(mods))
print(clasees.shape)
#print(Xd["QPSK",-20].shape)
# print(np.unique(clasees))a

'''
# Splitting the data into 50/50 training/testing ( first 50% )
limit = int(0.5*len(X))
X_train = X[0:limit, :]
Y_train = clasees[0:limit]
X_test = X[limit:, ]
Y_test = clasees[limit:]
print(len(X_train),"    ", len(X_test),"    ", len(Y_test),"    ",len(Y_train))
'''

# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,n_examples), size= int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
#changed from x to combined data
#X_train = combined_Data[train_idx]
#X_test =  combined_Data[test_idx]
X_train = X[train_idx]
X_test =  X[test_idx]
# one hot encoding for multiclass classification since there are 11 classes (11 modulation techniques)
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
# The Fully connected Neural Net as a baseline
in_shp = list(X_train.shape[1:])
# print(X_train.shape, "     ", in_shp)
print(in_shp+[1])
classes = mods

import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Automatically runs on GPU if one detected
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras



dr = 0.05
# Set up some params for training
nb_epoch = 100    # number of epochs to train on
batch_size = 1024  # training batch size

#fully connected neural network
'''
model = keras.models.Sequential()
model.add(Reshape(in_shp+[1], input_shape=in_shp))
model.add(Dropout(dr))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense2"))
model.add(Dropout(dr))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', name="dense3"))
model.add(Dropout(dr))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal', name="dense4"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense5" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
'''
'''
From Slides:
Before training a model, you need to configure the learning process, via the compile method. It
receives three arguments:
○ An optimizer. This could be the string identifier of an existing optimizer or an optimizer object.
○ A loss function. This is the objective that the model will try to minimize. It can be the string identifier of an
existing loss function or an objective function object.
○ A list of metrics. For any classification problem you will want to set this to metrics=['accuracy']. A metric
could be the string identifier of an existing metric or a custom metric function.
'''
#model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model1.summary()


#model1.fit(X_train, Y_train, epochs=nb_epoch , batch_size=batch_size, validation_split=0.05)
# perform training ...
#   - call the main training loop in keras for our network+dataset
'''filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history=model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
     validation_split=0.05,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model1.load_weights(filepath)

# Show simple version of performance
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print(model.metrics_names)
print(score)
'''

# CNN with PDF Architecture
model = keras.models.Sequential()
model.add(Reshape(in_shp+[1], input_shape=in_shp))
# no padding for the height and add padding to width (2 more columns)
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(64, (1, 3), padding='valid', activation='relu', name="conv1",
                  kernel_initializer='glorot_uniform', data_format="channels_last"))
# Adding dropout to inputs to next layer to avoid over fitting
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(16, (2, 3), padding='valid', activation='relu', name="conv2",
                  kernel_initializer='glorot_uniform', data_format="channels_last"))
model.add(Dropout(dr))
# The coming layer is dense to we need to flatten our inputs
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_normal', name="dense1"))
model.add(Dropout(dr))
model.add(Dense( len(classes), kernel_initializer='he_normal', name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(classes)]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

filepath = 'convmodrecnets_CNN2_0.5.wts.h5'
history=model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
     validation_split=0.05,
    callbacks = [
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    ])
# we re-load the best weights once training is finished
model.load_weights(filepath)

# Show simple version of performance
score = model.evaluate(X_test, Y_test, batch_size=batch_size)
print(model.metrics_names)
print(score)



'''# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2016)
n_examples = X.shape[0]
n_train = n_examples * 0.5
train_idx = np.random.choice(range(0,n_examples), size= int(n_train), replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx]
X_test =  X[test_idx]

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

print(len(X_train),"    ", len(X_test),"    ", len(Y_test),"    ",len(Y_train))
print(Y_test)

in_shp = list(X_train.shape[1:])
print (X_train.shape, in_shp)
classes = mods '''


'''
# Take first derivative in time for the raw features
# Derivative ( is the slope (diff in y / diff in x ) but diff in x-axis is = 1 ( 0-128 time instances) )
# Our derivative output size = sizeOfInput - 1 so we concatenate with 0 value to get the 128 size again.
X_arr = np.array(X)
print("Before derivative: ", X_arr.shape)
X_driv = np.diff(X_arr)
print("After: ", X_driv.shape)
z=np.zeros((220000, 2, 1))
print("shape before:")
print(X_driv.shape)
X_driv=np.concatenate((z, X_driv), axis=2)
print("shape after:")
print(X_driv.shape)
print("element with added zeros :")
print(X_driv[0])
print(X[0])
'''


'''import matplotlib.pyplot as plt

# print(X[0][0].shape)
# print(X[50][0])
# print(X[50][1])
y = np.zeros(shape=128)
for i in range(0, 128):
    y[i] = i
plt.plot(y, X[50][0])# size = 128
plt.plot(y, X[50][1])# size = 128
# print(mod_to_int)
plt.title((dataset_snrs[50], " ", mod_to_int[int(clasees[50])]))
plt.show()
'''
# print(snrs)