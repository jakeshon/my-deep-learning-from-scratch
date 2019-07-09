import numpy as np


def softmax(x):
    c = np.max(x)
    sum_exp_x = np.sum(np.exp(x-c))
    return np.exp(x-c)/sum_exp_x

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def relu(x):
    return np.maximum(0, x)

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entropy_error(y, t):
    if np.ndim(y) == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y+1e-7)) / batch_size