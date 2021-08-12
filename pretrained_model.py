
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
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dropout
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback

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

# For faster retrieval of data, using prefetch to optimize
# performance.
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Sweep config for wandb. 
sweep_config = {
    'name'  : "Surya_Pratik", 
    'method': 'grid', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },

    'parameters': {

        'modelname': {
            'values': ["resnet50", "inception_resnet_v2", "xception", "inception_v3", "mobilenetv2"]
        },
        'dropout_amount': {
            'values': [0.2, 0.3]
        },
        'base_learning_rate': {
            'values': [0.0001, 0.0005]
        },
        'ratio_finetune': {
            'values': [2, 3]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project = "dl_assignment-2_b-surya-pratik")

def mytrain():
    wandb.init(config = sweep_config)
    wandb.log({"images" : wandb.Image(plt)})
    config = wandb.config
    
    # We are about to use the xcpetion model.
    model_name = config.modelname
    
    # Amount of dropout being used.
    dropout_amount=config.dropout_amount
    
    # Learning rate
    base_learning_rate=config.base_learning_rate

    # What ratio of layers to fine-tune.
    ratio_finetune = config.ratio_finetune
    
    # Preprocess inputs for all the models.
    preprocess_inputs = {
        "inception_resnet_v2" : tf.keras.applications.inception_resnet_v2.preprocess_input,
        "xception" : tf.keras.applications.xception.preprocess_input,
        "inception_v3" : tf.keras.applications.inception_v3.preprocess_input,
        "resnet50" : tf.keras.applications.resnet50.preprocess_input,
        "mobilenetv2" : tf.keras.applications.mobilenet_v2.preprocess_input,
    }
    img_shape = image_size + (3,)

    # Load the base models.
    base_models = {
        "inception_resnet_v2" : tf.keras.applications.InceptionResNetV2(input_shape=img_shape, include_top=False, weights='imagenet'),
        "xception" : tf.keras.applications.Xception(input_shape=img_shape, include_top=False, weights='imagenet'),
        "inception_v3" : tf.keras.applications.InceptionV3(input_shape=img_shape, include_top=False, weights='imagenet'),
        "resnet50" : tf.keras.applications.ResNet50(input_shape=img_shape, include_top=False, weights='imagenet'),
        "mobilenetv2" : tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, weights='imagenet'),
    }
    
    base_model = base_models[model_name]
    preprocess_input = preprocess_inputs[model_name]
    
    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)
    
    # Trainable false since we are not performing any fine tuning.
    base_model.trainable = False
    
    # To convert the outputs as a single layer.
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    
    # One dense layer before the output layer.
    prediction_layer = tf.keras.layers.Dense(128, activation="relu")
    
    # Output layer of size 10.
    output_layer = tf.keras.layers.Dense(10, activation="softmax")
    
    # Data augmentation.
    data_augmentation = tf.keras.Sequential([
      tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
      tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    # init model
    inputs = tf.keras.Input(shape=img_shape)
    
    # Add data augmentation
    x = data_augmentation(inputs)
    
    # Preprocess the input
    x = preprocess_input(x)
    
    # Load the pre trained model.
    x = base_model(x, training=False)
    
    # Trim the output.
    x = global_average_layer(x)
    
    # Add dropout
    x = tf.keras.layers.Dropout(dropout_amount)(x)
    
    # Dense layer.
    x = prediction_layer(x)
    
    # Final output dense layer
    outputs = output_layer(x)
    
    # Model done
    model = tf.keras.Model(inputs, outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    epochs = 10
    # run = model.fit(train_ds, epochs = epochs, validation_data = val_ds, callbacks=[WandbCallback()])
    
    # Set all layers to be trainable initially for fine tuning.
    base_model.trainable = True
    
    # Set the cap layer from which to actually start fine tuning.
    fine_tune_at = int(len(base_model.layers)/ratio_finetune)
    
    # Set all the layers before fine_tune_at to be non trainable.
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
        
    model.compile(loss='categorical_crossentropy',optimizer = tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),metrics=['accuracy'])
    finetune_run = model.fit(train_ds, epochs = epochs, validation_data = val_ds, callbacks=[WandbCallback()])
    
wandb.agent(sweep_id, mytrain)