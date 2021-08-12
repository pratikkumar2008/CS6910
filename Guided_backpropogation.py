
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
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.preprocessing import image

# Image size is set as 150x150, batch size is taken as 32.
image_size = (150, 150)
batch_size = 32

# Load the test data. Here we are using 90% of the test data
# for training, and the rest 10% as validation.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "<path-to-nature_12K>/inaturalist_12K/train/",
        label_mode = 'categorical',
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "<path-to-nature_12K>/inaturalist_12K/val/",
    label_mode = 'categorical',
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Output has 10 classes.
num_classes = 10

# Input shape set.
input_shape = (150, 150, 3)

# Sweep config for the best trained model of all possible
# hyperparameter configurations
sweep_config = {
    'name'  : "Surya_Pratik", 
    'method': 'grid', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },

    'parameters': {

        'hidden_layer_size': {
            'values': [1024]
        },
        'layer_1_size': {
            'values': [128]
        },
        'layer_2_size': {
            'values': [64]
        },
        'layer_3_size': {
            'values': [64]
        },
        'layer_4_size': {
            'values': [64]
        },
        'layer_5_size': {
            'values': [32]
        },
        'epochs': {
            'values': [25]
        },
        'dropout': {
            'values': [0.3]
        },
        'activation': {
            'values': ['relu']
        },
        'batch_normalisation': {
            'values': ['false']
        },
        'data_augmentation': {
            'values': ['<null>']
        }

    }
}

# Function to de-process the image
def deprocess_image(x):
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Making a custom gradient for relU.
@tf.custom_gradient
def guidedReLU(x):
    def grad(dy):
        return tf.cast(dy>0, "float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad


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

    # Consider the output of the CONV5 layer of our network.
    new_model = keras.Model(inputs = model.inputs, outputs = model.get_layer("max_pooling2d_4").output)
    
    # Use the custom made relU
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation==tf.keras.activations.relu:
            layer.activation = guidedReLU


    for img, label in test_ds.take(1):
        img_array = tf.convert_to_tensor(img, dtype = 'float32')
        # Computing and plotting
        plt.figure(figsize=(64, 64))
        for i in range(10):
            with tf.GradientTape() as tape:
                inp = tf.expand_dims(img_array[22], 0)
                tape.watch(inp)
                out = new_model(inp)[0,:,:,2*i]
            gradients = tape.gradient(out, inp)[0]
            plt.subplot(10, 1, i+1)
            plt.title(f"w.r.t neuron number {2*i+1}")
            plt.imshow(np.flip(deprocess_image(np.array(gradients)),-1))
        
    wandb.log({"Images" : plt})
    plt.show()

wandb.agent(sweep_id, mytrain)