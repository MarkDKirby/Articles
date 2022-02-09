#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 21:31:43 2021

@author: markkirby
"""

#import all libraries
import itertools
import warnings
import random
import shutil
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
warnings.simplefilter(action='ignore', category=FutureWarning)


    
#set paths to get the pictures
train_path = '/Users/markkirby/Downloads/archive/forky-dataset/train'
valid_path = '/Users/markkirby/Downloads/archive/forky-dataset/valid'
test_path = '/Users/markkirby/Downloads/archive/forky-dataset/test'

#Get the pictures from the data set and process them to work with VGG16
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224,224), classes=['Fork', 'Knife', "Spoon"], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224,224), classes=['Fork', 'Knife', "Spoon"], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224,224), classes=['Fork', 'Knife', "Spoon"], batch_size=10, shuffle=False)


assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 3

imgs, labels = next(train_batches)


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(8,4))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    print(type(fig))
    
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True
    """
    plt.imshow(cm,interpolation='nearest')
    plt.title(title)
    df_cm = pd.DataFrame(cm, index = [i for i in ["Fork","Knife","Spoon"]], columns = [i for i in ["Fork", "Knife", "Spoon"]])
    plt.figure(figsize = (10,7))
    sns.heatmap(df_cm, annot=True,cmap="Blues")
    
    if normalize:
        cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Normalized Confusion Matrix")
        
    else:
        print('Confusion Matrix, Without Normalization')
    
    print(cm)
    
    thresh = cm.max()/2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, cm[i,j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i,j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

def count_params(model):
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_weights])
    return {'non_trainable_params':non_trainable_params, 'trainable_params': trainable_params}
    

vgg16_model = tf.keras.applications.vgg16.VGG16()

params = count_params(vgg16_model)
assert params['non_trainable_params']==0
assert params['trainable_params'] == 138357544
print("It works so far...")

test_imgs, test_labels = next(test_batches)
print(test_labels)


print(test_batches.classes)

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
    
    
model.add(Dense(units=4096))
model.add(Dense(units=3, activation='softmax'))
model.summary()

params = count_params(model)


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

predictions = model.predict(x= test_batches, verbose = 0)

predictions_rounded = np.round(predictions)

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['Fork','Knife','Spoon']

plot_confusion_matrix(cm,classes=cm_plot_labels, normalize=False, title= "Confusion Matrix")

print(test_batches.class_indices)


###########
# THE END #
###########