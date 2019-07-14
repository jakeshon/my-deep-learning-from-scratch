import numpy as np


class SGD:

    def __init__(self, lr):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]



class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.momentum = momentum
        self.lr = lr
        self.v = None

    def update(self, params, grads):
        if self.v == None:
            for key, value in params.items():
                self.v = {}
                self.v[key] = np.zeros_like(value)
            
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]

class AdaGrad:
    def __init__(self, lr=0.01):
        self.h = None
        self.lr = lr

    def update(self, params, grads):
        if self.h == None:
            self.h = {}
            for key,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

