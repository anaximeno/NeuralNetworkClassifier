#!/usr/bin/env python3
# %load main.py
"""I made this when studying on the web book Neural Networks and Deep Learning [http://neuralnetworksanddeeplearning.com/] and 
some videos made by Andrew Ng on Youtube [https://www.youtube.com/channel/UCcIXc5mJsHVYTZR1maL5l9w].

This python program is a Neural Network classifier for the mnist digit dataset using vanilla Neural Networks.

`Example`: You may run this program on a jupyter notebook or a python console prompt, and type for example:
    >>> fit([784, 30, 10], train, 20, 10, 0.5, 0.9, test_data=test, \
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
    name = 'quadratic'

    @staticmethod
    def fn(a, y, z):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(a, y, z):
        return (a-y)*sigmoid_prime(z)


class CrossEntropyCost(CostFunction):
    name = 'crossentropy'

    @staticmethod
    def fn(a, y, z):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(a, y, z):
        return (a-y)

class NeuralNetwork(object):

    def __init__(self, sizes: list, cost: CostFunction = CrossEntropyCost, weight_initializer: WeightInitializer = None):
        self.num_layers = len(sizes)
        self.cost = cost
        self.sizes = sizes
        self.input_size = sizes[0]
        self.output_size = sizes[-1]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if weight_initializer is not None:
            self.weights = weight_initializer(sizes).get_weights()
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
 #   def __call__(self, *args: Any, **kwds: Any) -> Any:
  #      return super().

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
                cp = self.evaluate(test_data)
                print(f"Epoch {j+1}/{epochs}:\n Test Accuracy = {(100*cp/n_test):.2f}%")
                # print(f" Test Loss = {loss/n_test}")
            else:
                print(f"Epoch {j+1} Complete.")

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        m = len(mini_batch)

        X, Y = zip(*mini_batch)
        X = np.array(X).reshape(m, self.input_size,).T
        Y = np.array(Y).reshape(m, self.output_size,).T

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

        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        # loss = sum(self.cost.fn(a, y, None) for a, y in test_results)
        correct_predictions = sum(int(a == y) for a, y in test_results)

        return correct_predictions

    def save(self):

        obj = dict(
            net='VectorizedNetwork',
            sizes=self.sizes,
            cost=self.cost.name,
            weights=[self.weights, self.biases]
        )

        with open("export.pkl", "wb") as exp:
            pickle.dump(obj, exp)


class LoadNet(NeuralNetwork):

    def __init__(self, network_path: str):
        self.loaded_from = network_path

        with open(network_path, 'rb') as  f:
            dt = pickle.load(f)
        
        if dt['cost'] == CrossEntropyCost.name:
            cost = CrossEntropyCost
        elif dt['cost'] == QuadraticCost.name:
            cost = QuadraticCost
        else:
            print(f"error: {dt['cost']} was not recognized!");
            exit(1)

        super(LoadNet, self).__init__(sizes=dt['sizes'], cost=cost)
        self.weights, self.biases = dt['weights']


def fit(sizes, training_data, epochs, mini_batch_size, lr, lmbda, test_data=None, save_net=False,
    validation_data=None, weight_initializer=None, costfunction=CrossEntropyCost):
    
    print('\n * Training the Network *\n')

    net = NeuralNetwork(sizes, costfunction, weight_initializer)
    net.SGD(training_data, epochs, mini_batch_size, lr, lmbda, test_data=test_data)
  
    if validation_data:
        n = len(validation_data)
        cp = net.evaluate(validation_data)
        print(f"\nValidation Set:\n Accuracy = {(100*cp/n):.2f}%")

    return net

if __name__ == '__main__':
    # uncommet the code below for stopping warnings
    # import warnings
    # warnings.filterwarnings('ignore')
    
    # load the training, validation/dev and testing data which will be used for training the model
    train, dev, test = [list(data) for data in mloader.load_data_wrapper('./mnist.pkl.gz')]
    
    try:
        # The input and output size are allready determined by default, so you just have to determine the hidden sizes
        sizes = eval(input('> Hidden Sizes : '))
        epochs = int(input('> Epochs : '))
        mbs = int(input('> Mini Batch Size : '))
        lr = float(input('> Learning Rate : '))
    except ValueError:
        print("error: wrong types: Sizes must be a sequence of units of the layers!")
        exit(1)
    
    #### FIXME: Must fix cross entropy loss

    
    if isinstance(sizes, int):
        sizes = [784, sizes, 10]
    else:
        sizes = [784]+list(sizes)+[10]

    model = fit(
        costfunction=QuadraticCost,
        sizes=sizes,
        training_data=train,
        epochs=epochs,
        mini_batch_size=mbs,
        lr=lr,
        lmbda=0.9,
        test_data=test,
        validation_data=dev,
        weight_initializer=GoldenWeightInitializer,
        # save_net=True    # uncomment to save the model!
    )

    # If you saved the model
    # save_model = LoadNet('export.pkl')
    # Next do what you want with the model
    # ...

