# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:29:57 2023

@author: ragas
"""

root_path = 'C:/Users/ragas/OneDrive/Pictures/Desktop/MajorDataset'

# Local variables


# Datasets
train_ds = root_path + '/data/new_datasets/train'
test_ds = root_path + '/data/new_datasets/test'
valid_ds = root_path + '/data/new_datasets/validation'

# Checkpoint for effnet
checkpoints_effnet = root_path + '/checkpoints/effnet/' + 'bestcheckpoint_{epoch:02d}_{val_loss:.3f}_{val_accuracy:.3f}.hdf5'
best_weights_effnet = root_path + '/checkpoints/effnet/'


# Import necessary libs

import pandas as pd
import os
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import gc

# Find and delete duplicated images
if False:
    
  sum = 0
  dir = ''
  for filename in tqdm(os.listdir(dir)):
      if (filename.find("(")) != -1:
          os.remove(dir + "/" + filename)
          sum += 1
  sum
  


# Separate melanoma jpg files from other
if False:
    
  for filename in os.listdir(others_images_path):
      id = os.path.splitext(filename)[0]
      for i in melanoma_data.index:
          if melanoma_data['image_id'][i] == id:
              shutil.move(others_images_path + '/' + filename, melanoma_images_path)

# Count jpg files in directory
if False:

  dir = ''
  sum = 0
  for filename in os.listdir(dir):
    if filename.endswith(".jpg"): 
        sum = sum + 1
  print(sum)
  
# Count melanoma jpg files in directory
if False:

  dir = ''
  sum = 0
  for filename in os.listdir(dir):
      id = os.path.splitext(filename)[0]
      for i in melanoma_data.index:
          if melanoma_data['image_id'][i] == id:
              sum = sum + 1
  print(sum)
  
# Create balanced datasets with data augmentation
if False:

  # To generate names for the images
  def formatter(prefix, suffix):
    if suffix < 1000:
      prefix = prefix + '0'
    if suffix < 100:
      prefix = prefix + '0'
    if suffix < 10:
      prefix = prefix + '0'
    return prefix + str(suffix) + ".jpg"

  # Melanoma images
  dir = root_path + '/data/images/melanoma'
  dst = root_path + '/data/new_images/melanoma/'
  index = 0
  for filename in tqdm(os.listdir(dir)):
    src = dir + "/" + filename
    if index < 111:
       shutil.copy(src, dst + formatter("melanoma_", index))
    elif index < 222:
       shutil.copy(src, dst + formatter("melanoma_", index))
    else:
      # Rotate images
       img = cv2.imread(src)
       img_horizontal = cv2.flip(img, 0)
       img_vertical = cv2.flip(img, 1)
       img_horizontal_vertical = cv2.flip(img, -1)
       # Save images
       cv2.imwrite(dst + formatter("melanoma_", index), img)
       cv2.imwrite(dst + formatter("melanoma_", index+1), img_horizontal)
       cv2.imwrite(dst + formatter("melanoma_", index+2), img_vertical)
       cv2.imwrite(dst + formatter("melanoma_", index+3), img_horizontal_vertical)
       index += 3
    index += 1

  # Others images
  dir = root_path + '/images/others'
  prefix = root_path + '/data/new_images/others/'
  index = 0
  for filename in tqdm(os.listdir(dir)[:3786]):
      src = dir + "/" + filename
      shutil.copy(src, prefix + formatter("others_", index))
      index += 1
      
# Separate melanoma images
if False:

  dir = root_path + '/data/new_images/melanoma'
  dst_root = root_path + '/data/new_datasets/'
  index = 0

  for filename in tqdm(os.listdir(dir)):
    src = dir + "/" + filename
    if index < 111:
        shutil.copy(src, dst + "test/melanoma/")
    elif index < 222:
        shutil.copy(src, dst + "validation/melanoma/")
    else:
        shutil.copy(src, dst + "train/melanoma/")
    index += 1
    
# Separate others images
if False:
    
  dir = root_path + '/data/new_images/others'
  dst_root = root_path + '/data/new_datasets/'
  index = 0

  for filename in tqdm(os.listdir(dir)):
    src = dir + "/" + filename
    if index < 111:
        shutil.copy(src, dst + "test/others/")
    elif index < 222:
        shutil.copy(src, dst + "validation/others/")
    else:
        shutil.copy(src, dst + "train/others/")
    index += 1
    
# Import necessary libs

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.regularizers import l2
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


from keras import layers
import efficientnet.keras as efn
from keras import regularizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras import applications

# Set 0 if you want to use EfficientNet based model.
# Set 1 if you want to use InceptionV3 based mofel.
model_id = 0

#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#------------------------------------------- LOAD DATA -------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
# Load data in batches with an ImageDataGenerator

datagen_train = ImageDataGenerator(
                    featurewise_center=False, 
                    samplewise_center=False,
                    featurewise_std_normalization=False, 
                    samplewise_std_normalization=False,
                    zca_whitening=False, 
                    rotation_range=40,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    brightness_range=None, 
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=0.0, 
                    fill_mode='nearest', 
                    cval=0.0, 
                    horizontal_flip=False,
                    vertical_flip=False,
                    rescale=1.0/255.0, 
                    preprocessing_function=None,
                    data_format=None, 
                    validation_split=0.0, 
                    dtype=None)

datagen_test = ImageDataGenerator(
                    rescale=1.0/255.0)

# Shape of the images (lxl)
l = 224


# Iterators for each dataset
train_it = datagen_train.flow_from_directory(train_ds, class_mode='binary', batch_size=16, target_size= (l, l))
val_it = datagen_test.flow_from_directory(valid_ds, class_mode='binary', batch_size=16, target_size= (l, l))
test_it = datagen_test.flow_from_directory(test_ds, class_mode='binary',batch_size=16, target_size=(l, l))

batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

# Build the model

base_model = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

model = base_model.output
model = GlobalAveragePooling2D()(model)
model = Dense(128, activation='relu')(model)
predictions = Dense(1, activation='sigmoid')(model)
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the layers
for layer in model.layers[0:]:
    layer.trainable = True

# Set learning rate

my_lr = 0.0001


optimizer=optimizers.Adam(lr=my_lr)
model.compile(optimizer=optimizer,loss="binary_crossentropy", metrics=['accuracy']) 

# Visualizing the model 

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Callback functions

# Earlystop
earlystop_callback = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# Save the most accurate model's weights
cp = checkpoints_effnet

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                          filepath=cp,
                          save_weights_only=True,
                          monitor='val_accuracy',
                          mode='max',
                          save_best_only=True)

# Modify LR during train
lr_callback = ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=0.000005)

# Train the model

my_callbacks = [
                earlystop_callback,
                checkpoint_callback,
                # lr_callback
]

h = model.fit(train_it, epochs = 50, validation_data = val_it, callbacks = my_callbacks, shuffle = True, steps_per_epoch = 100)

# Load the model

dir = best_weights_effnet


if len(os.listdir(dir)) == 0:
  print("There are no previous checkpoints. Make sure you use the correct directory.")
else:
  filename = os.listdir(dir)[-1]
  best_weight = dir + filename
  print(best_weight)
  model.load_weights(best_weight)
  
# Evaluate the model 

results = model.evaluate(test_it, batch_size=16)
print("Test loss: {:.4f}".format(results[0]*100))
print("Test accuracy: {:.4f}".format(results[1]*100))


from keras.models import load_model
model.save("newmodel.h5")
loaded_model = load_model("newmodel.h5")


# Plotting train loss and validation loss

plt.figure()
plt.plot(h.history['loss'],color="olive", label='loss')
plt.plot(h.history['val_loss'], color="plum", label='validation loss')
plt.legend(loc='best')
plt.title(label="Loss and Validation Loss")

# Plotting train accuracy and validation accuracy

plt.figure()
plt.plot(h.history['accuracy'],color="mediumspringgreen", label='accuracy')
plt.plot(h.history['val_accuracy'], color="black", label='validation accuracy')
plt.legend(loc='best')
plt.title(label="Accuracy and Validation Accuracy")