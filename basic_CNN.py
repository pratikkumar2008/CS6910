
###########################################################
###                                                     ###
###     TITLE : CS6910 : Assignment - 2                 ###
###     COLLABORATORS : SUDHEENDRA - CS18B006           ###
###                     PRATIK KUMAR - EE20M018         ###
###     DATE OF SUBMISSION : 15 - 04 - 21               ###
###                                                     ###
###########################################################


# Importing the necessary modules and libraries
from tensorflow.keras.layers import Conv2D,Activation,MaxPooling2D,Flatten,Dense
from tensorflow.keras import Sequential
from tensorflow.keras import preprocessing
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.layers import BatchNormalization
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np

# Image size is set as 150x150, batch size is taken as 32.
image_size = (150, 150)
batch_size = 32


# Load the training data. Here we are using 90% of the training data
# for training, and the rest 10% as validation.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pratik/Downloads/cs6910/inaturalist_12K/train/",
    validation_split = 0.1,
    subset="training",
    label_mode = 'categorical',
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    
)

# Load the validation data as 10% of the test data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "/home/pratik/Downloads/cs6910/inaturalist_12K/train/",
    validation_split = 0.1,
    subset="validation",
    label_mode = 'categorical',
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)


# For faster retrieval of data, using prefetch to optimize
# performance.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Output has 10 classes.
num_classes = 10

# Input shape set.
input_shape = (150, 150, 3)

# Default hyperparamters to build a CNN.
defaults = dict(
    hidden_layer_size=128,
    layer_1_size=64,
    layer_2_size=64,
    layer_3_size=64,
    layer_4_size=64,
    layer_5_size=64,
    activation='relu',
    epochs=25,
    size_l1=3,
    size_l2=3,
    size_l3=3,
    size_l4=3,
    size_l5=3,
    max_pool=2
)

# Define a sequential model
model = Sequential()

# Adding the first convolution layer.
model.add(Conv2D(defaults["layer_1_size"], (defaults["size_l1"], defaults["size_l1"]), input_shape=input_shape))
model.add(Activation(defaults["activation"]))
model.add(MaxPooling2D(pool_size=(defaults["max_pool"], defaults["max_pool"])))

# Adding the second convolution layer.
model.add(Conv2D(defaults["layer_2_size"], (defaults["size_l2"], defaults["size_l2"])))
model.add(Activation(defaults["activation"]))
model.add(MaxPooling2D(pool_size=(defaults["max_pool"], defaults["max_pool"])))

# Adding the third convolution layer.
model.add(Conv2D(defaults["layer_3_size"], (defaults["size_l3"], defaults["size_l3"])))
model.add(Activation(defaults["activation"]))
model.add(MaxPooling2D(pool_size=(defaults["max_pool"], defaults["max_pool"])))

# Adding the fourth convolution layer.
model.add(Conv2D(defaults["layer_4_size"], (defaults["size_l4"], defaults["size_l4"])))
model.add(Activation(defaults["activation"]))
model.add(MaxPooling2D(pool_size=(defaults["max_pool"], defaults["max_pool"])))

# Adding the fifth convolution layer.
model.add(Conv2D(defaults["layer_5_size"], (defaults["size_l5"], defaults["size_l5"])))
model.add(Activation(defaults["activation"]))
model.add(MaxPooling2D(pool_size=(defaults["max_pool"], defaults["max_pool"])))

# Flatten the model 
model.add(Flatten())

# Added one more dense layer before the final output layer.
model.add(Dense(defaults["hidden_layer_size"], activation="relu"))

# Final dense layer with output layer of size 10.
model.add(Dense(10, activation="softmax"))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_ds,epochs=defaults["epochs"], validation_data=val_ds)

