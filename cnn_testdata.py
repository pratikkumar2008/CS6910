
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
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

# Image size is set as 150x150, batch size is taken as 32.
image_size = (150, 150)
batch_size = 32

# Output has 10 classes.
num_classes = 10

# Input shape set.
input_shape = (150, 150, 3)

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

# Loading the test data
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
#train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
#val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# The best performing hyperparameter configuration
defaults = dict(
    hidden_layer_size=1024,
    layer_1_size=128,
    layer_2_size=64,
    layer_3_size=64,
    layer_4_size=64,
    layer_5_size=32,
    activation='relu',
    epochs=25,
    dropout=0.3
)

# Init model
inputs = keras.Input(shape=input_shape)

# Adding the first convolution layer.
x=Conv2D(defaults["layer_1_size"], (3, 3))(inputs)

# Perform Batch normalization
x=BatchNormalization()(x)
x=Activation(defaults["activation"])(x)
x=MaxPooling2D(pool_size=(2, 2))(x)

# Add dropout
x=Dropout(defaults["dropout"])(x)

# Adding the second convolution layer.
x=Conv2D(defaults["layer_2_size"], (3, 3))(x)

# Perform Batch normalization
x=BatchNormalization()(x)
x=Activation(defaults["activation"])(x)
x=MaxPooling2D(pool_size=(2, 2))(x)

# Add dropout
x=Dropout(defaults["dropout"])(x)

# Adding the third convolution layer.
x=Conv2D(defaults["layer_3_size"], (3, 3))(x)

# Perform Batch normalization
x=BatchNormalization()(x)
x=Activation(defaults["activation"])(x)
x=MaxPooling2D(pool_size=(2, 2))(x)

# Add dropout
x=Dropout(defaults["dropout"])(x)

# Adding the fourth convolution layer.
x=Conv2D(defaults["layer_4_size"], (3, 3))(x)

# Perform Batch normalization
x=BatchNormalization()(x)
x=Activation(defaults["activation"])(x)
x=MaxPooling2D(pool_size=(2, 2))(x)

# Add dropout
x=Dropout(defaults["dropout"])(x)

# Adding the fifth convolution layer.
x=Conv2D(defaults["layer_5_size"], (3, 3))(x)

# Perform Batch normalization
x=BatchNormalization()(x)
x=Activation(defaults["activation"])(x)
x=MaxPooling2D(pool_size=(2, 2))(x)

# Add dropout
x=Dropout(defaults["dropout"])(x)

# Flatten the model 
x=Flatten()(x)

# Added one more dense layer before the final output layer.
x=Dense(defaults["hidden_layer_size"], activation="relu")(x)

# Final dense layer with output layer of size 10.
outputs=Dense(10, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="surya-pratik-model")

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(train_ds,epochs=defaults["epochs"], validation_data=val_ds)
results = model.evaluate(test_ds)
print(results)

# Class names
class_names = train_ds.class_names

# Plotting
plt.figure(figsize=(3, 10))

for images, labels in test_ds.take(5):
	labels=model.predict(images)
	for i in range(30):
		label = np.argmax(labels[i])
		ax = plt.subplot(3, 10, i + 1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(class_names[label])
		plt.axis("off")

plt.show()


for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# get filter weights
	filters, biases = layer.get_weights()
	print(layer.name, filters.shape)
	break

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

n_filters, ix = defaults["layer_1_size"], 1
for i in range(n_filters):
	f = filters[:, :, :, i]
	ax = plt.subplot(16, 8, ix)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.imshow(f[:, :, 0], cmap='gray')
	ix += 1
plt.show()
