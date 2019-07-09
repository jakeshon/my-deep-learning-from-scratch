import numpy as np

#%%
def softmax(x):
    c = np.max(x)
    sum_exp_x = np.sum(np.exp(x-c))
    return np.exp(x-c)/sum_exp_x



#%%
