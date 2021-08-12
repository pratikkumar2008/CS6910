
DL_Assignment3.ipynb :- Questions 1, 2, 3, 4
--------------------------------------------------------------------------------------------

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

        'latent_dim': {
            'values': [128]
        },
        'epochs': {
            'values': [10]
        },
        'num_encoder': {
            'values': [1]
        },
        'num_decoder': {
            'values': [1]
        },
        'n_type': {
            'values': ['lstm']
        },
        'dropout': {
            'values': [null']
        }
        
    }
}

The above configuration build a LSTM with latent dim as 128, 1 encoder and 1 decoder model,
with no dropout and trains it for 10 epochs.

We have stored the best model out of all hyperparameter configurations as a.h2s.
To use this model elsewhere, use model = keras.models.load_model("a.h2s")


attention.ipynb :- Questions 5
--------------------------------------------------------------------------------------------

No of units are specified as parameter to class BahdanauAttention.
The input(sentence), output(result) and the plot can be obtained as -  
result, sentence, attention_plot = evaluate(sentence)
