
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
    "<path-to-nature_12K>/inaturalist_12K/train/",
    validation_split = 0.1,
    subset="training",
    label_mode = 'categorical',
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    
)

# Load the validation data as 10% of the test data
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "<path-to-nature_12K>/inaturalist_12K/train/",
    validation_split = 0.1,
    subset="validation",
    label_mode = 'categorical',
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "<path-to-nature_12K>/inaturalist_12K/val/",
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

# Sweep config for wandb
sweep_config = {
    'name'  : "Surya_Pratik", 
    'method': 'grid', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },

    'parameters': {

        'hidden_layer_size': {
            'values': [512,1024]
        },
        'layer_1_size': {
            'values': [64,128,256]
        },
        'layer_2_size': {
            'values': [32,64,128]
        },
        'layer_3_size': {
            'values': [32,64,128]
        },
        'layer_4_size': {
            'values': [32,64,128,256]
        },
        'layer_5_size': {
            'values': [32,64]
        },
        'epochs': {
            'values': [15,25,50]
        },
        'activation': {
            'values': ['sigmoid','relu']
        },
        'dropout': {
            'values': [0.2,0.3,0.4]
        },
        'batch_normalisation': {
            'values': ['true','<null>']
        },
        'data_augmentation': {
            'values': ['true','<null>']
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project = "dl_assignment2-surya-pratik-temp")




def mytrain():
    wandb.init(config = sweep_config)
    wandb.log({"images" : wandb.Image(plt)})
    config = wandb.config

    # Init model
    inputs = keras.Input(shape=input_shape)

    # Adding the first convolution layer.
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])
    if(config.data_augmentation=='true'):
        x = data_augmentation(inputs)
        x=Conv2D(config.layer_1_size, (3, 3))(x)
    else:
        x=Conv2D(config.layer_1_size, (3, 3))(inputs)   
        
    if(config.batch_normalisation=='true'):
        # Perform Batch normalization
        x=BatchNormalization()(x)
    x=Activation(config.activation)(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add dropout
    x=Dropout(config.dropout)(x)

    # Adding the second convolution layer.
    x=Conv2D(config.layer_2_size, (3, 3))(x)
    if(config.batch_normalisation=='true'):
        # Perform Batch normalization
        x=BatchNormalization()(x)
    x=Activation(config.activation)(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add dropout
    x=Dropout(config.dropout)(x)

    # Adding the third convolution layer.
    x=Conv2D(config.layer_3_size, (3, 3))(x)
    if(config.batch_normalisation=='true'):
        # Perform Batch normalization
        x=BatchNormalization()(x)
    x=Activation(config.activation)(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add dropout
    x=Dropout(config.dropout)(x)

    # Adding the fourth convolution layer.
    x=Conv2D(config.layer_4_size, (3, 3))(x)
    if(config.batch_normalisation=='true'):
        # Perform Batch normalization
        x=BatchNormalization()(x)
    x=Activation(config.activation)(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add dropout
    x=Dropout(config.dropout)(x)

    # Adding the fifth convolution layer.
    x=Conv2D(config.layer_5_size, (3, 3))(x)
    if(config.batch_normalisation=='true'):
        # Perform Batch normalization
        x=BatchNormalization()(x)
    x=Activation(config.activation)(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    
    # Add dropout
    x=Dropout(config.dropout)(x)

    # Flatten the model 
    x=Flatten()(x)

    # Added one more dense layer before the final output layer.
    x=Dense(config.hidden_layer_size, activation="relu")(x)

    # Final dense layer with output layer of size 10.
    outputs=Dense(10, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="surya-pratik-model")

    model.summary()
    model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])

    # Using categorical crossentropy as the loss, and adam as the optimizer.
    model.fit(train_ds,epochs=config.epochs, validation_data=val_ds,callbacks=[WandbCallback()])
	
	
wandb.agent(sweep_id, mytrain)