import numpy as np
import common.functions as cf


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        A1 = np.dot(x, W1) + b1
        Z1 = cf.sigmoid(A1)
        A2 = np.dot(Z1, W2) + b2
        Y = cf.softmax(A2)

        return Y

    def loss(self, x, t):
        y = self.predict(x)
        loss = cf.cross_entropy_error(y, t)
        return loss

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = cf.numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = cf.numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = cf.numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = cf.numerical_gradient(loss_W, self.params['b2'])
        return grads

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
x = np.random.rand(100, 784)
t = np.random.rand(100, 10)
