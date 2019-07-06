import numpy as np

def softmax(x):
    sum_exp_x = np.sum(np.exp(x))
    return np.exp(x)/sum_exp_x