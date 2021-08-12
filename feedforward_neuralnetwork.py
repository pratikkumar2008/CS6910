###########################################################
###                                                     ###
###     TITLE : CS6910 : Assignment - 1                 ###
###     COLLABORATORS : SUDHEENDRA - CS18B006           ###
###                     PRATIK KUMAR - EE20M018         ###
###     DATE OF SUBMISSION : 14 - 03 - 21               ###
###                                                     ###
###########################################################

# REPORT LINK : https://wandb.ai/pratikkumar2008/dl_assignment-surya-pratik/reports/Final-Report-of-Surya-Pratik--Vmlldzo1Mjg0NTE

# Get all the necessary libraries
import wandb
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import copy, math

# Load the fashion_mnist data.
mnist = fashion_mnist

# Get the data into local variables.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Resize the images from 28x28 to 784x1, and normalize each value to be from 0 to 1. 
x_train = x_train.reshape(60000, 784)/256
x_test = x_test.reshape(10000, 784)/256

# Helper lists to print sample images from each class.
uniqueData = []
uniqueLabels = []
uniqueLablesDes = []

# All the class names from the fashion_mnist datasets.
class_names=['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

for i in range(len(y_train)):
    if y_train[i] not in uniqueLabels and len(uniqueLabels) < 10:
        uniqueData.append(x_train[i])
        uniqueLabels.append(y_train[i])

# True label names for each of the class.
label = {0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}

for i in range(len(uniqueLabels)):
    uniqueLablesDes.append(label[uniqueLabels[i]])

# Plot the images.
for i in range(len(uniqueLabels)):
    plt.subplot(2,10, 2*i+1)
    plt.imshow(uniqueData[i])
    plt.title(uniqueLablesDes[i])
    
        
    
# Sigmoid function.
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Derivate of the sigmoid function.
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Derivative of tanh function.
def d_tanh(x):
    return 1 - np.square(np.tanh(x))

# Activation function - using Sigmoid, tanh.
def g(ai, num_neurons, activation_func):
    if activation_func == "sigmoid":
        hi = np.empty([num_neurons, 1])
        for i in range(num_neurons):
            hi[i] = sigmoid(ai[i])
        return hi
    elif activation_func == "tanh":
        hi = np.empty([num_neurons, 1])
        for i in range(num_neurons):
            hi[i] = np.tanh(ai[i])
        return hi

# Derivate of the activation function - of Sigmoid, and of tanh.
def g1(ai, num_neurons, activation_func):
    if activation_func == "sigmoid":
        hi = np.empty([num_neurons, 1])
        for i in range(num_neurons):
            hi[i] = d_sigmoid(ai[i])
        return hi
    elif activation_func == "tanh":
        hi = np.empty([num_neurons, 1])
        for i in range(num_neurons):
            hi[i] = d_tanh(ai[i])
        return hi

# Output function - getting the probability distributions.
def o(al, num_output):
    output = np.empty([num_output, 1])
    s = 0
    for i in range(num_output):
        s += np.exp(al[i])
    for i in range(num_output):
        output[i] = np.exp(al[i]) / s
    return output

# Using Random Distributions to set random values to W's and B's.
def init(W, B, num_input, num_hlayers, num_neurons, num_output):

    W[1] = np.random.normal(0, np.sqrt(1/(num_input+num_neurons)),(num_neurons, num_input))
    B[1] = np.zeros([num_neurons, 1])
    for i in range(2, num_hlayers):
        W[i] = np.random.normal(0, np.sqrt(1/(num_neurons+num_neurons)),(num_neurons, num_neurons))
        B[i] = np.zeros([num_neurons, 1])
    W[num_hlayers] = np.random.normal(0, np.sqrt(1/(num_neurons+num_output)),(num_output, num_neurons))
    B[num_hlayers] = np.zeros([num_output, 1])

# Code for forward propogation. Returns the final probability distribution for each of the label.
def forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func):
    H[0] = x_train[ii].reshape(-1, 1)
    for i in range(1, num_hlayers):
        A[i] = np.dot(W[i], H[i-1]) + B[i]
        H[i] = g(A[i], num_neurons, activation_func)
    A[num_hlayers] = np.dot(W[num_hlayers], H[num_hlayers-1]) + B[num_hlayers]
    fin_ans = o(A[num_hlayers], num_output)
    return fin_ans

# Helper function which returns an array with 1 at index i, all other elements being 0 of length len.
def e(i, len):
    x = np.zeros([len, 1])
    x[i] = 1
    return x

# Code for back propogation. Returns the partial derivatives of Loss function w.r.t each of the W's and B's.
def back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func):
    da = {}
    dh = {}
    dal = np.empty([num_output,1])
    if loss_func == "cross_entropy":
        dal = fin_ans - e(y_train[ii], num_output)
    elif loss_func == "squared_error":
        for i in range(num_output):
            dal[i] = np.dot(np.transpose(np.multiply(fin_ans, fin_ans - e(y_train[ii], num_output))), e(i, num_output) - np.ones([num_output,1]) * fin_ans[i])
    da[num_hlayers] = dal
    for k in range(num_hlayers, 0, -1):
        if k in dw.keys():
            dw[k] += np.dot(da[k],np.transpose(H[k-1]))
            db[k] += da[k]
        else:
            dw[k] = np.dot(da[k],np.transpose(H[k-1]))
            db[k] = da[k]
        dh[k-1] = np.dot(np.transpose(W[k]), da[k])
        if k != 1:
            da[k-1] = np.multiply(dh[k-1], g1(A[k-1], num_neurons, activation_func).reshape(-1, 1))
    return dw, db

# Find the loss for a given probability distribution, loss function.
def loss(fin_ans, y_train, loss_func, ii, num_output):
    if loss_func == "cross_entropy":
        return -np.log(fin_ans[y_train[ii]])
    elif loss_func == "squared_error":
        return np.square(fin_ans - e(y_train[ii], num_output))

# Find the accuracy (Both validation accuracy and the test accuracy), for the current W and B configuration.
def accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func):
    acc = 0
    y_pred = np.zeros(y_test.shape)
    for ii in range(10000):
        A = {}
        H = {}
        H[0] = x_test[ii].reshape(-1, 1)
        for i in range(1, num_hlayers):
            A[i] = np.dot(W[i], H[i-1]) + B[i]
            H[i] = g(A[i], num_neurons, activation_func)
        A[num_hlayers] = np.dot(W[num_hlayers], H[num_hlayers-1]) + B[num_hlayers]
        fin_ans = o(A[num_hlayers], num_output)
        index = 0
        val = 0
        for i in range(np.shape(fin_ans)[0]):
            if val < fin_ans[i]:
                index = i
                val = fin_ans[i]
        y_pred[ii] = index
        acc += int(index == y_test[ii])
    print("Acc is ", acc/100, "%")
    wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs = None, preds = y_pred, y_true = y_test, class_names = class_names)})
    wandb.log({"test_acc": acc / 100})
    val_acc = 0
    total_val_loss = 0 
    for ii in range(54000, 60000):
        A = {}
        H = {}
        fin_ans = forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func)
        val_loss = loss(fin_ans, y_train, loss_func, ii, num_output)
        total_val_loss += val_loss
        index = 0
        val = 0
        for i in range(np.shape(fin_ans)[0]):
            if val < fin_ans[i]:
                index = i
                val = fin_ans[i]
        val_acc += int(index == y_train[ii])
    print("Validation Acc is ", val_acc/60, "%")
    print("Total Validation Loss is ", total_val_loss)
    wandb.log({"val_acc" : val_acc/60})
    wandb.log({"val_loss" : total_val_loss})

# Function to complete training of the dataset. Can pass most of the hyperparameters as function arguments.
def train(x_train, y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, learning_rate, num_epochs, batch_size, opt, train_set, activation_func, loss_func):
    init(W, B,num_input, num_hlayers, num_neurons, num_output)
    if opt == "sgd":
        for inp in range(num_epochs):
            dw = {}
            db = {}
            tot_loss = 0
            for ii in range(train_set):
                fin_ans = forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func)
                curr_loss = loss(fin_ans, y_train, loss_func, ii, num_output)
                tot_loss += curr_loss
                back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func)
                if ii % batch_size == 0:
                    for i in range(1, num_hlayers+1):
                        W[i] -= learning_rate * dw[i]

                    for i in range(1, num_hlayers):    
                        B[i] -= learning_rate * db[i]
                    dw = {}
                    db = {}
            wandb.log({"epoch": inp})
            wandb.log({"loss": tot_loss})
            accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func)
    elif opt == "mbgd":
        prev_w, prev_b, gamma = copy.deepcopy(W), copy.deepcopy(B), 0.9
        for inp in range(num_epochs):
            dw = {}
            db = {}
            tot_loss = 0
            for ii in range(train_set):
                fin_ans = forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func)
                curr_loss = loss(fin_ans, y_train, loss_func, ii, num_output)
                tot_loss += curr_loss
                back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func)
                if ii % batch_size == 0:
                    for i in range(1, num_hlayers+1):
                        v_w = gamma * prev_w[i] * int(inp != 0) + learning_rate * dw[i]
                        W[i] -= v_w
                        prev_w[i] =  copy.deepcopy(v_w)
                    for i in range(1, num_hlayers):    
                        v_b = gamma * prev_b[i] * int(inp != 0) + learning_rate * db[i]
                        B[i] -= v_b
                        prev_b[i] = copy.deepcopy(v_b)
                    dw = {}
                    db = {}
            wandb.log({"epoch": inp})
            wandb.log({"loss": tot_loss})
            accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func)
    elif opt == "nagd":
        prev_w, prev_b, gamma = copy.deepcopy(W), copy.deepcopy(B), 0.9
        for inp in range(num_epochs):
            dw = {}
            db = {}
            tot_loss = 0
            for ii in range(train_set):
                if ii % batch_size == 0:
                    for i in range(1, num_hlayers+1):
                        v_w = gamma * prev_w[i] * int(inp != 0)
                        W[i] -= v_w
                    for i in range(1, num_hlayers):    
                        v_b = gamma * prev_b[i] * int(inp != 0)
                        B[i] -= v_b
                fin_ans = forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func)
                curr_loss = loss(fin_ans, y_train, loss_func, ii, num_output)
                tot_loss += curr_loss
                back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func)
                if ii % batch_size == 0:
                    for i in range(1, num_hlayers+1):
                        v_w = gamma * prev_w[i] * int(inp != 0) + learning_rate * dw[i]
                        W[i] -= v_w
                        prev_w[i]=  copy.deepcopy(v_w)
                    for i in range(1, num_hlayers):    
                        v_b = gamma * prev_b[i] * int(inp != 0) + learning_rate * db[i]
                        B[i] -= v_b
                        prev_b[i] = copy.deepcopy(v_b)
                    dw = {}
                    db = {}
            wandb.log({"epoch": inp})
            wandb.log({"loss": tot_loss})
            accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func)
    elif opt == "rmsprop":
        beta, eps, beta1 = 0.1, 1e-8, 0.9
        v_w = {}
        v_b = {}
        for inp in range(num_epochs):
            dw = {}
            db = {}
            tot_loss = 0
            for ii in range(train_set):
                fin_ans = forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func)
                curr_loss = loss(fin_ans, y_train, loss_func, ii, num_output)
                tot_loss += curr_loss
                back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func)
                if ii % batch_size == 0:
                    for i in range(1, num_hlayers+1):
                        if(inp==0):
                            v_w[i] = np.zeros(dw[i].shape, dtype=float)
                            v_b[i] = np.zeros(db[i].shape, dtype=float)

                    for i in range(1, num_hlayers+1):
                        v_w[i] = beta1 * v_w[i] + (1 - beta) * np.square(dw[i])
                        W[i] -= (learning_rate / np.sqrt(v_w[i] + eps)) * dw[i]

                    for i in range(1, num_hlayers):
                        v_b[i] = beta1 * v_b[i] + (1 - beta) * np.square(db[i])
                        B[i] -= (learning_rate / np.sqrt(v_b[i] + eps)) * db[i]
                    dw = {}
                    db = {}
            wandb.log({"epoch": inp})
            wandb.log({"loss": tot_loss})
            accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func)
    elif opt == "adam":
        m_w={}
        v_w={}
        m_b={}
        v_b={}
        eps, beta1, beta2 = 1e-8, 0.9, 0.999
        for inp in range(num_epochs):
            dw = {}
            db = {}
            tot_loss = 0
            for ii in range(train_set):
                fin_ans = forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func)
                curr_loss = loss(fin_ans, y_train, loss_func, ii, num_output)
                tot_loss += curr_loss
                back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func)
                if ii % batch_size == 0:
                    for i in range(1, num_hlayers+1):
                        if(inp == 0):
                            m_w[i] = np.zeros(dw[i].shape, dtype=float)
                            v_w[i] = np.zeros(dw[i].shape, dtype=float)
                            m_b[i] = np.zeros(db[i].shape, dtype=float)
                            v_b[i] = np.zeros(db[i].shape, dtype=float)
                    for i in range(1, num_hlayers+1):
                        m_w[i] = beta1 * m_w[i] + (1 - beta1) * dw[i]
                        m_w_cap = m_w[i] / (1 - math.pow(beta1, (ii / batch_size) + 1))
                        v_w[i] = beta2 * v_w[i] + (1 - beta2) * np.square(dw[i])
                        v_w_cap = v_w[i] / (1 - math.pow(beta2, (ii / batch_size) + 1))   
                        W[i] -= (learning_rate / np.sqrt(v_w_cap + eps)) * m_w_cap

                    for i in range(1, num_hlayers):
                        m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
                        m_b_cap = m_b[i] / (1-math.pow(beta1,(ii / batch_size) + 1))
                        v_b[i] = beta2 * v_b[i] +(1 - beta2) * np.square(db[i])
                        v_b_cap = v_b[i] / (1-math.pow(beta2, (ii / batch_size) + 1))
                        B[i] -= (learning_rate / np.sqrt(v_b_cap + eps)) * m_b_cap
                    dw = {}
                    db = {}
            wandb.log({"epoch": inp})
            wandb.log({"loss": tot_loss})
            accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func)
    elif opt == "nadam":
        eps, beta1, beta2= 1e-8, 0.9, 0.999
        m_w = {}
        v_w = {}
        m_b = {}
        v_b = {}
        for inp in range(num_epochs):
            dw = {}
            db = {}
            tot_loss = 0
            for ii in range(train_set):
                fin_ans = forward_prop(x_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, ii, activation_func)
                curr_loss = loss(fin_ans, y_train, loss_func, ii, num_output)
                tot_loss += curr_loss
                dw, db = back_prop(y_train, num_hlayers, num_neurons, num_input, num_output, W, B, A, H, dw, db, ii, fin_ans, activation_func, loss_func)
                if ii % batch_size == 0:
                    for i in range(1, num_hlayers+1):
                        if(inp == 0):
                            m_w[i] = np.zeros(dw[i].shape, dtype=float)
                            v_w[i] = np.zeros(dw[i].shape, dtype=float)
                            m_b[i] = np.zeros(db[i].shape, dtype=float)
                            v_b[i] = np.zeros(db[i].shape, dtype=float)
                    for i in range(1, num_hlayers+1):
                        m_w[i] = beta1 * m_w[i] + (1 - beta1) * dw[i]
                        m_w_cap = m_w[i] / (1 - math.pow(beta1, (ii / batch_size) + 1))
                        v_w[i] = beta2 * v_w[i] + (1 - beta2) * np.square(dw[i])
                        v_w_cap = v_w[i] / (1 - math.pow(beta2,(ii / batch_size) + 1))
                        W[i] -= (learning_rate / np.sqrt(v_w_cap + eps)) * (beta1 * m_w_cap + ((1 - beta1) / (1 - math.pow(beta1,(ii / batch_size) + 1)) * dw[i])) 
                    for i in range(1, num_hlayers):
                        m_b[i] = beta1 * m_b[i] + (1 - beta1) * db[i]
                        m_b_cap = m_b[i] / (1 - math.pow(beta1, (ii / batch_size) + 1))
                        v_b[i] = beta2 * v_b[i] + (1 - beta2) * np.square(db[i])
                        v_b_cap = v_b[i] / (1 - math.pow(beta2, (ii / batch_size) + 1))
                        B[i] -= (learning_rate / np.sqrt(v_b_cap + eps)) * (beta1 * m_b_cap + ((1 - beta1) / (1 - math.pow(beta1,(ii / batch_size) + 1)) * db[i]))
                    dw = {}
                    db = {}
            wandb.log({"epoch": inp})
            wandb.log({"loss": tot_loss})
            accuracy(x_test, y_test, x_train, y_train, W, B, num_hlayers, num_output, num_neurons, activation_func)
            
# Sweep config for wandb plotting
sweep_config = {
    'name'  : "Surya_Pratik", 
    'method': 'random', 
    'metric': {
      'name': 'val_acc',
      'goal': 'maximize'   
    },

    'parameters': {

        'num_hlayers': {
            'values': [3,5]
        },
        'num_epochs': {
            'values': [10,15]
        },
        'num_neurons': {
            'values': [32,64]
        },
        'learning_rate': {
            'values': [1e-2,5e-3,1e-3]
        },
        'batch_size': {
            'values': [32,64]
        },
        'opt': {
            'values': ["sgd","mbgd","nagd","rmsprop","adam","nadam"]
        },
        'activation_func': {
            'values': ["tanh","sigmoid"]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project = "dl_assignment-surya-pratik")

# Set values manually for the current configuration of the dataset
num_output = 10
num_input = 784
train_set = 54000

W = {}
B = {}
A = {}
H = {}
loss_func = "cross_entropy"


def mytrain():
    wandb.init(config = sweep_config)
    wandb.log({"images" : wandb.Image(plt)})
    config = wandb.config
    train(x_train, y_train, config.num_hlayers, config.num_neurons, num_input, num_output, W, B, A, H, config.learning_rate, config.num_epochs, config.batch_size, config.opt, train_set, config.activation_func, loss_func)

wandb.agent(sweep_id, mytrain)
