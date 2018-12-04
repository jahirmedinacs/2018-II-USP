
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from copy import copy as copy

import os
import sys

import pprint

from keras.datasets import mnist


# In[2]:


PLOT = True
NUM_OF_CLASS = 10
VERBOSE = True


# In[3]:


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

import keras.utils


# In[4]:


# The data, split between train and test sets:
(Input_train, Objetive_train), (Input_test, Objetive_test) = mnist.load_data()

Input_train = Input_train.reshape(tuple(list(Input_train.shape) + [1]))
Input_test = Input_test.reshape(tuple(list(Input_test.shape) + [1]))

print('Input_train shape:', Input_train.shape)
print(Input_train.shape[0], 'train samples')
print(Input_test.shape[0], 'test samples')


# In[5]:


# Convert class vectors to binary class matrices.
Objetive_train = keras.utils.to_categorical(Objetive_train, NUM_OF_CLASS)
Objetive_test = keras.utils.to_categorical(Objetive_test, NUM_OF_CLASS)


# In[6]:


#Architecture
def createModel(input_shape, nClasses):
    model = Sequential()
    
    model.add(Conv2D(25, (2, 2), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(20, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))
    
#     model.add(Conv2D(48, (2, 2), padding='same', activation='selu'))
#     model.add(Conv2D(48, (4, 4), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.3))
    
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='selu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.3))
 
    model.add(Flatten())
#     model.add(Dense(500, activation='selu'))
#     model.add(Dropout(0.8))
#     model.add(Dense(300, activation='relu'))
#     model.add(Dropout(0.4))
    model.add(Dense(nClasses, activation='softmax'))
     
    return model


# In[7]:


## Train Parameters

epochs = 50
batch_size = 1000

###

output_model = createModel(Input_train.shape[1:], NUM_OF_CLASS)
output_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])


# In[8]:


output_model.summary()


# In[9]:


history = output_model.fit(Input_train, Objetive_train, batch_size=batch_size, epochs=epochs, verbose=VERBOSE, 
                       validation_data=(Input_test, Objetive_test))


# In[10]:


output_model.evaluate(Input_test, Objetive_test)


# In[11]:


if PLOT:
    ## Plots
    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'.r')
    plt.plot(history.history['val_loss'],'.b')
    plt.plot(history.history['loss'],'--y')
    plt.plot(history.history['val_loss'],'--g')
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()

    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'.r')
    plt.plot(history.history['val_acc'],'.b')
    plt.plot(history.history['acc'],'--y')
    plt.plot(history.history['val_acc'],'--g')
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()


# In[12]:


output_model.save(filepath="./mnist_model", include_optimizer=1)

