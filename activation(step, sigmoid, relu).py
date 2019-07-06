import numpy as np
import matplotlib.pyplot as plt

#%%
def step_function(x):
    y = x > 0
    return y.astype(np.int)

#%%
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()




#%%
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)

#%%
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x,y)




#%%
