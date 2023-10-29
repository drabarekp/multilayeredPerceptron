import numpy as np


def linear(x):
    return x


def linear_derivative(x):
    return 1


def sigmoid(x):
    #
    if min(x) < -128:
        allowed = np.full(x.shape, -128)
        x = np.maximum(x, allowed)
    if max(x) > 128:
        allowed = np.full(x.shape, 128)
        x = np.minimum(x, allowed)
    return np.exp(-np.logaddexp(0, -x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    if isinstance(x, np.ndarray):
        return np.array([0 if x[i] < 0 else x[i] for i in range(len(x))])
    else:
        return 0 if x < 0 else x


def relu_derivative(x):
    if isinstance(x, np.ndarray):
        return np.array([0 if x[i] < 0 else 1 for i in range(len(x))])
    else:
        return 0 if x < 0 else 1


def arctan(x):
    return np.arctan(x)


def arctan_derivative(x):
    return 1 / (x ** 2 + 1)


def mean_squared_error(network_output, expected_output):
    if len(network_output) != len(expected_output):
        raise Exception(f'different input array lengths: {len(network_output)}, {len(expected_output)}')
    return ((network_output - expected_output) ** 2).mean()


def mean_squared_error_derivative(network_output, expected_output):     # with respect to network output
    if len(network_output) != len(expected_output):
        raise Exception(f'different input array lengths: {len(network_output)}, {len(expected_output)}')
    return 2 * (network_output - expected_output) / len(network_output)   # changed order of subtraction


def mean_absolute_error(network_output, expected_output):
    if len(network_output) != len(expected_output):
        raise Exception(f'different input array lengths: {len(network_output)}, {len(expected_output)}')
    return np.absolute(network_output - expected_output).sum() / len(network_output)


def mean_absolute_error_derivative(network_output, expected_output):    # with respect to network output
    if len(network_output) != len(expected_output):
        raise Exception(f'different input array lengths: {len(network_output)}, {len(expected_output)}')
    return np.array([-1 / len(network_output) if network_output[i] < expected_output[i] else
                     1 / len(network_output) if network_output[i] > expected_output[i] else 0
                     for i in range(len(network_output))])
