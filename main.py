from math import e

import matplotlib.pyplot as plt
import numpy as np


def main():
    # func = sigmoid
    # func = tanh
    # func = relu
    func = softplus

    x = np.linspace(-5, 5, 100)
    y, y_prime = func(x)

    plt.plot(x, y)
    plt.show()

    plt.plot(x, y_prime)
    plt.show()


def sigmoid(x):
    fx = 1.0 / (1.0 + e ** (-x))
    dfdx = (1.0 - fx) * fx
    return fx, dfdx


def tanh(x):
    fx = (e ** x - e ** (-x)) / (e ** x + e ** (-x))
    dfdx = 4.0 / (e ** x + e ** (-x)) ** 2
    return fx, dfdx


def relu(x):
    fx = x * (x > 0)
    dfdx = 1 * (x > 0)
    return fx, dfdx


def softplus(x):
    fx = np.log(1 + np.exp(x))
    dfdx = 1.0 / (1.0 + np.exp(-x))
    return fx, dfdx


if __name__ == "__main__":
    main()
