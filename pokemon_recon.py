import numpy as np
import pandas as pd


print("Goodbye World")

data = pd.read_csv("/data/numeron/mnist_train.csv")

print(data.head())

data = np.array(data)
m, n = data.shape



np.random.shuffle(data)

data_t = data[0:10000].T
Y_tag = data_t[0]
X_tag = data_t[1:n]

data_train = data[10000:m].T

print(data_train[0:5, 0:5])
Y_train = data_train[0]
X_train = data_train[1:n]


def init_params():
    W1 = np.random.rand(10,784)
    b1 = np.random.rand(10,1)
    W2 = np.random.rand(10,10)
    b2 = np.random.rand(10,1)
    return W1,b1, W2, b2

def ReLU(Z):
    return np.maximum(0,Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1 , Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def back_prop(Z1,A1,Z2,A2,W2,W1,Y):
    one_hot_Y = one_hot(Y)
    

