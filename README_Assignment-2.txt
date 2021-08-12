
###################################### PART - A ########################################

convolition_neuralnetwork.py :- Questions 1, 2, 3
--------------------------------------------------------------------------------------------

The functions and various variables defined and their respective uses are as follows : 

VARIABLES :

img_size : The size of the image being used. Type : Tuple.

batch_size : The size of each batch. Type : INT.

train_ds : The training dataset. Type : Keras dataset

val_ds : The validation dataset. Type : Keras dataset

num_classes : Num of classes in the output layer. Type : INT

input_shape : Shape of input given to the model. Type : Tuple.


FUNCTIONS : 

mytrain() :- Takes no parameters, makes the model and runs it on the training and
    validation data.

To run the code, simply set the values to each of the variables inside sweep_config . Then run 
python convolition_neuralnetwork.py to execute the code.
An example configuration is given below : -

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
            'values': [64]
        },
        'epochs': {
            'values': [25]
        },
        'activation': {
            'values': ['relu']
        },
        'dropout': {
            'values': [0.2]
        },
        'batch_normalisation': {
            'values': ['true']
        },
        'data_augmentation': {
            'values': ['<null>']
        }
    }
}
The above configuration assumes the size of the first layer as 128, and the next 4 layers of sizes 64.
Total number of epochs = 25, with dropout as 20%, relu activation function and using batch normalization.


cnn_testdata.py :- Question 4
--------------------------------------------------------------------------------------------

The functions and various variables defined and their respective uses are as follows : 

VARIABLES :

img_size : The size of the image being used. Type : Tuple.

batch_size : The size of each batch. Type : INT.

train_ds : The training dataset. Type : Keras dataset

val_ds : The validation dataset. Type : Keras dataset

test_ds : The test dataset. Type : Keras dataset

num_classes : Num of classes in the output layer. Type : INT

input_shape : Shape of input given to the model. Type : Tuple.


FUNCTIONS : 

No function -


Our best model as acquired from the previous questions in Part-A is as follows -
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

We are using this model to test the data.To run the code, run
python cnn_testdata.py


Guided_backpropogation.py :- Question 5
--------------------------------------------------------------------------------------------

The functions and various variables defined and their respective uses are as follows : 

VARIABLES :

img_size : The size of the image being used. Type : Tuple.

batch_size : The size of each batch. Type : INT.

train_ds : The training dataset. Type : Keras dataset

num_classes : Num of classes in the output layer. Type : INT

input_shape : Shape of input given to the model. Type : Tuple.


FUNCTIONS : 

mytrain() :- Takes no parameters, makes the model and runs it on the training and
    validation data.

deprocess_image() :- Used to normalize, clip and conver the image. Takes the image
    as argument. Returns a de-processed image.

guidedReLU() :- The modified version of ReLU. Guided backpropogation is done using this
    activation function

Our best model is given as :-

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

We are running guided backpropogation on this model. To run the code, run
python Guided_backpropogation.py



###################################### PART - B ########################################

pretrained_model.py :- Question 1, 2, 3
--------------------------------------------------------------------------------------------

The functions and various variables defined and their respective uses are as follows : 

VARIABLES :

img_size : The size of the image being used. Type : Tuple.

batch_size : The size of each batch. Type : INT.

train_ds : The training dataset. Type : Keras dataset

num_classes : Num of classes in the output layer. Type : INT

input_shape : Shape of input given to the model. Type : Tuple.

model_name : Name of the model we want to load and use. Type : String.

preprocess_input : This is a dictionary containing preprocessed inputs for all the models
    we want to use. Type : Dictionary

base_model : This is a dictionary containing  all the models we want to use. Type : Dictionary

ratio_finetune : What ratio of layers to finetune. Type : INT.

FUNCTIONS : 

mytrain() :- Takes no parameters, makes the model and runs it on the training and
    validation data.

To run the code, simply set the values to each of the variables inside sweep_config . Then run 
python pretrained_model.py to execute the code.
An example configuration is given below : -

sweep_config = {
    'name'  : "Surya_Pratik", 
    'method': 'grid', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },

    'parameters': {

        'modelname': {
            'values': ["resnet50"]
        },
        'dropout_amount': {
            'values': [0.2]
        },
        'base_learning_rate': {
            'values': [0.0001]
        },
        'ratio_finetune': {
            'values': [2]
        }
    }
}