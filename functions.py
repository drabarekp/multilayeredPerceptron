import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
    # return x


def mean_squared_error(a, b):
    return ((a - b) ** 2).mean()


# not sure if correct
def mean_squared_error_derivative_a(a, b):
    if len(a) != len(b):
        raise Exception(f"vector lengths in mse: {len(a)}, {len(b)}")
    return 2 * (a - b) / len(a)

