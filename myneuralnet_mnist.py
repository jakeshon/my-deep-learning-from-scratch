# coding: utf-8

from dataset.mnist import load_mnist
import pickle
import numpy as np
from sigmoid import sigmoid
from softmax import softmax

# 데이터 가져오는 부분
def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist()
    return x_test, t_test

# 학습된 weight 가져오는 부분
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network

# 예측
def predict(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    B1, B2, B3 = network['b1'], network['b2'], network['b3']

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
    A3 = np.dot(Z2, W3) + B3
    Y = softmax(A3)

    return Y

x_test, t_test = get_data()
network = init_network()

batch_size = 100
acc_count = 0

for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    acc_count += np.sum(p == t_test[i:i+batch_size])

#accarr = accarr.astype(np.uint8)
acc = acc_count/len(x_test)
print("Accuracy :" + str(acc))



