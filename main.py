#!/usr/bin/env python3
# %load main.py
"""I made this when studying on the web book Neural Networks and Deep Learning [http://neuralnetworksanddeeplearning.com/] and 
some videos made by Andrew Ng on Youtube [https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w].

This python Program is a Neural Network classifier for the mnist digit dataset.

`Example`: You may run this program on a jupyter notebook or a python console prompt, and type for example:
    >>> train([784, 30, 10], train, 20, 10, 0.5, 0.9, test_data=test, \
        save_net=True, validation_data=dev, weight_initializer=GoldenWeightInitializer)

Anaxímeno Brito.
"""
import numpy as np
import random
import pickle
import os
import sys
import pdb

import mnist_loader as mloader


def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


class CostFunction:
    @staticmethod
    def fn(a, y, z):
        pass

    @staticmethod
    def delta(a, y, z):
        pass


class WeightInitializer:
    def __init__(self, sizes):
        self.sizes = sizes

    def get_weights(self):
        pass

class NormalWeightInitializer(WeightInitializer):

    def get_weights(self):
        return [
            np.random.randn(y, x)*np.sqrt(1.0/x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]


class GoldenWeightInitializer(WeightInitializer):

    def get_weights(self):
        gr = (1 + 5**0.5)/2
        return [
            np.random.randn(y, x)*np.sqrt(gr/x) for x, y in zip(self.sizes[:-1], self.sizes[1:])
        ]


class QuadraticCost(CostFunction):

    @staticmethod
    def fn(a, y, z):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(a, y, z):
        return (a-y)*sigmoid_prime(z)


class CrossEntropyCost(CostFunction):

    @staticmethod
    def fn(a, y, z):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y, z):
        return (a-y)

class NeuralNetwork(object):

    def __init__(self, sizes, cost: CostFunction = CrossEntropyCost, weight_initializer: WeightInitializer = None):
        self.num_layers = len(sizes)
        self.cost = cost
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if weight_initializer is not None:
            self.weights = weight_initializer(sizes).get_weights()
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, lmbda=0, test_data=None):
        """Stochastic Gradient Descent Algorithm."""
        n = len(training_data)
        if test_data:
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            for k in range(0, n, mini_batch_size):
                self.update_mini_batch(
                    mini_batch=training_data[k:k+mini_batch_size],
                    eta=eta,
                    lmbda=lmbda,
                    n=n
                )
            if test_data:
                print(
                    f"Epoch {j}: Test Results ~ {(100*self.evaluate(test_data)/n_test):.2f}%")
            else:
                print(f"Epoch {j} Complete.")

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        m = len(mini_batch)

        X, Y = zip(*mini_batch)
        X = np.array(X).reshape(m, 784,).T
        Y = np.array(Y).reshape(m, 10,).T

        delta_b, delta_w = self.backprop(X, Y, m)
        # Using L2 Regularization(weight decay)
        self.weights = [(1 - eta*(lmbda/n))*w - (eta/m)*dw for w, dw in zip(self.weights, delta_w)]
        self.biases = [b - (eta/m)*db for b, db in zip(self.biases, delta_b)]

    def backprop(self, X, Y, m):
        """Vectorized backpropagation method, it calculates
        for all training inputs on the mini_batch at same time."""
        DB = [np.zeros(b.shape) for b in self.biases]
        DW = [np.zeros(w.shape) for w in self.weights]

        # feedforward/forward pass
        A = X
        As = [X]  # list to store all the activations, layer by layer
        Zs = []  # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            Z = np.dot(w, A)+b
            Zs.append(Z)
            A = sigmoid(Z)
            As.append(A)

        # backward pass
        DZ = (self.cost).delta(As[-1], Y, Zs[-1])

        DW[-1] = (1.0/m)*np.dot(DZ, As[-2].T)
        DB[-1] = (1.0/m)*np.sum(DZ, axis=1, keepdims=True)
        for l in range(2, self.num_layers):
            Sp = sigmoid_prime(Zs[-l])
            DZ = np.dot(self.weights[-l+1].T, DZ) * Sp
            DB[-l] = (1.0/m)*np.sum(DZ, axis=1, keepdims=True)
            DW[-l] = (1.0/m)*np.dot(DZ, As[-l-1].T)
        return DB, DW

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for x, y in test_results)

    def save(self):
        with open("export.pkl", "wb") as p:
            obj = dict(net='VectorizedNetwork', sizes=self.sizes, weights=self.weights, biases=self.biases)
            pickle.dump(obj, p)

    def load(self, filename: str):
        with open(filename, "rb") as f:
            dt = pickle.load(f)

        assert dt['sizes'] == self.sizes, f"Sizes Must be Equal, Inference data sizes: {dt['sizes']}"
        self.weights = dt['weights']
        self.biases = dt['biases']

    def predict(self, x):
        return np.argmax(self.feedforward(x))


def fit(sizes, training_data, epochs, mini_batch_size, lr, lmbda, test_data=None, save_net=False,
    validation_data=None, weight_initializer=None, costfunction=CrossEntropyCost):
    net = NeuralNetwork(sizes, costfunction, weight_initializer)
    net.SGD(training_data, epochs, mini_batch_size, lr, lmbda, test_data=test_data)
    if validation_data:
        print("Validation Results: ", net.evaluate(validation_data))

    if save_net is True:
        net.save()


if __name__ == '__main__':
    # load the train, validation/dev and test data to be used on the execution of the programm
    train, dev, test = [list(data) for data in mloader.load_data_wrapper('./mnist.pkl.gz')]

    # Now you have to call the train function with all variables!
    # fit(...)
