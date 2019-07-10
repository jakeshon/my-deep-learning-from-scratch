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

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return f(x+h)-f(x-h)/(2*h)

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val
    
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        x -= lr * numerical_gradient(f, x)
    
    return x